#!/usr/bin/env python3
"""
OCT Structured Report Generator (GPT-5, order-legend segmentation) + ROI output

Generates a strict JSON object with:
- disease_class      (echoed EXACTLY from your input)
- layers_table       (7 rows: NFL..RPE; status in {normal, thinned, elevated, disrupted, irregular})
- biomarker_table    (0–4 rows: name, location, relevance; include ONLY if clearly visible)
- clinical_summary   (4–5 sentences, evidence-based, no alternative diagnoses)
- roi_list           (0–8 items; per-biomarker regions with normalized coordinates)

ROI rules
- ROIs are only emitted for biomarkers that are clearly visible.
- Each ROI references a biomarker via biomarker_ref (index into biomarker_table).
- Coordinates are normalized to the referenced image size (values in [0,1]).
  * box:  coords  = [x, y, w, h]
  * poly: points  = [[x,y], [x2,y2], ...] (>= 3 points)
- target chooses which image coordinate space applies to:
  "original" (raw OCT), "seg" (segmentation), or "both".

Security
- Do NOT hardcode API keys in source code.
- This script reads the OpenAI API key from the environment variable OPENAI_API_KEY.
"""

import base64
import json
import mimetypes
import os
from typing import Optional, Dict, Any, List

from openai import OpenAI


ALLOWED_DISEASE = ["DME", "Drusen", "CNV", "Normal"]
ALLOWED_LAYERS = ["NFL", "GCL/IPL", "INL", "OPL", "ONL", "PRL", "RPE"]
ALLOWED_STATUS = ["normal", "thinned", "elevated", "disrupted", "irregular"]
ALLOWED_ROI_TYPES = ["box", "poly"]
ALLOWED_TARGETS = ["original", "seg", "both"]

ORDER_LEGEND_TEXT = (
    "Segmentation order (top→bottom): NFL, GCL/IPL, INL, OPL, ONL, PRL, RPE. "
    "Top = inner retina; bottom = outer retina/RPE interface."
)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY environment variable. "
            "Set it before running this script."
        )
    return OpenAI(api_key=api_key)


def encode_image_to_base64(path: str) -> str:
    """Read an image file and return a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_data_mime(path: str) -> str:
    """
    Infer a MIME type for an image file.

    Falls back to a simple extension-based mapping when mimetypes cannot infer.
    """
    mt, _ = mimetypes.guess_type(path)
    if mt:
        return mt
    ext = os.path.splitext(path.lower())[1]
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


def build_prompt(disease_class: str) -> str:
    """
    Construct the instruction prompt that enforces:
    - exact disease_class echo
    - strict tool/function output only
    - ordered layers_table
    - biomarkers only when clearly visible
    - ROIs only when confidently localizable
    """
    return f"""
You are a retina-specialized medical vision-language model.

Inputs:
1) disease_class = "{disease_class}"  (echo this exact string; do not change or infer a different disease)
2) An OCT B-scan (raw).
3) An OCT image with retinal layers segmented by ORDER (not by color).
   {ORDER_LEGEND_TEXT}

Return ONLY the final result via the function "submit_oct_report" with EXACTLY these keys:
- disease_class: must equal "{disease_class}" exactly.
- layers_table: list of 7 objects in this exact order ["NFL","GCL/IPL","INL","OPL","ONL","PRL","RPE"].
  Each: {{"layer":"<one of the 7>","status":"<normal|thinned|elevated|disrupted|irregular>"}}.
  Use "normal" if no clear abnormality is visible.
- biomarker_table: 0–4 objects; include a biomarker only if clearly visible in the images. If none are visible, return [].
  Each biomarker: {{"name":"<seen biomarker>","location":"<specific site/layer>","relevance":"<1–2 sentence clinical importance in the context of disease_class>"}}
- clinical_summary: one paragraph of 4–5 sentences describing only visible findings, per-layer status highlights,
  and any visible biomarkers. Do not name or imply any disease other than "{disease_class}". If features seem atypical,
  report the visible facts but do not propose alternatives.
- roi_list: 0–8 objects highlighting visible biomarkers only. If a biomarker cannot be localized with confidence, omit its ROI.
  Each ROI object must be:
  {{
    "biomarker_ref": <integer index into biomarker_table>,
    "name": "<short ROI label matching the biomarker>",
    "type": "<box|poly>",
    "target": "<original|seg|both>",
    "layer": "<one of ['NFL','GCL/IPL','INL','OPL','ONL','PRL','RPE'] or '' if N/A>",
    "confidence": <0.0–1.0>,
    "coords": [<x>, <y>, <w>, <h>],
    "points": [[<x>,<y>], ...]
  }}
  Rules for ROIs:
  - Use normalized coordinates (0–1).
  - Provide either 'coords' (box) or 'points' (poly), not both.
  - Create at most one ROI per biomarker. If localization is uncertain, skip the ROI.

Strict evidence rules:
- Use only the provided images; never invent features.
- Prefer segmentation ordering for layer boundaries/status.
- Use the raw B-scan for fluid, PED, SHRM, drusen, cysts, hyperreflective foci, etc.
- If a biomarker is not clearly visible, omit it and omit its ROI.
- Output must be strict JSON in the function arguments: no prose, no markdown, no comments.
""".strip()


def build_tools_schema(disease_class: str) -> List[Dict[str, Any]]:
    """
    Tool schema that:
    - locks disease_class to the exact input using enum: [disease_class]
    - constrains layers and statuses to allowed sets
    - enforces a strict JSON shape (additionalProperties=False)
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_oct_report",
                "description": "Return the OCT report strictly as a structured object.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "disease_class": {"type": "string", "enum": [disease_class]},
                        "layers_table": {
                            "type": "array",
                            "minItems": 7,
                            "maxItems": 7,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "layer": {"type": "string", "enum": ALLOWED_LAYERS},
                                    "status": {"type": "string", "enum": ALLOWED_STATUS},
                                },
                                "required": ["layer", "status"],
                            },
                        },
                        "biomarker_table": {
                            "type": "array",
                            "minItems": 0,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "location": {"type": "string"},
                                    "relevance": {"type": "string"},
                                },
                                "required": ["name", "location", "relevance"],
                            },
                        },
                        "clinical_summary": {"type": "string"},
                        "roi_list": {
                            "type": "array",
                            "minItems": 0,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "biomarker_ref": {"type": "integer", "minimum": 0},
                                    "name": {"type": "string"},
                                    "type": {"type": "string", "enum": ALLOWED_ROI_TYPES},
                                    "target": {"type": "string", "enum": ALLOWED_TARGETS},
                                    "layer": {"type": "string", "enum": ALLOWED_LAYERS + [""]},
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                    "coords": {
                                        "type": "array",
                                        "minItems": 4,
                                        "maxItems": 4,
                                        "items": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                        },
                                    },
                                    "points": {
                                        "type": "array",
                                        "minItems": 3,
                                        "items": {
                                            "type": "array",
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "items": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
                                            },
                                        },
                                    },
                                },
                                "required": ["biomarker_ref", "name", "type", "target", "confidence"],
                            },
                        },
                    },
                    "required": [
                        "disease_class",
                        "layers_table",
                        "biomarker_table",
                        "clinical_summary",
                        "roi_list",
                    ],
                },
            },
        }
    ]


def _extract_structured_json(resp) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from:
    - modern tool_calls
    - legacy function_call (dict or object)

    Returns the parsed dict if found, else None.
    """
    try:
        msg = resp.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if getattr(tc, "type", "") == "function":
                    fn = getattr(tc, "function", None)
                    if fn and getattr(fn, "name", "") == "submit_oct_report":
                        args = getattr(fn, "arguments", "") or ""
                        if isinstance(args, str) and args.strip():
                            return json.loads(args)

        fc = getattr(msg, "function_call", None)
        if isinstance(fc, dict) and fc.get("name") == "submit_oct_report":
            args = fc.get("arguments") or ""
            if isinstance(args, str) and args.strip():
                return json.loads(args)

        if hasattr(msg, "function_call") and not isinstance(fc, dict):
            fco = getattr(msg, "function_call")
            if getattr(fco, "name", "") == "submit_oct_report":
                args = getattr(fco, "arguments", "") or ""
                if isinstance(args, str) and args.strip():
                    return json.loads(args)

    except Exception as e:
        print("[extract] Failed to parse structured tool output:", e)

    return None


def _in_01(x: float) -> bool:
    """Return True if x is a number in [0, 1]."""
    try:
        return 0.0 <= float(x) <= 1.0
    except Exception:
        return False


def validate_and_warn(report: Dict[str, Any]) -> None:
    """
    Validate the report structure and print warnings (does not raise).

    Checks:
    - disease_class is valid
    - layers_table has 7 rows in the exact required order
    - biomarker_table shape and required keys
    - clinical_summary is non-empty
    - roi_list shape, coordinates/points constraints, and biomarker_ref bounds
    """
    problems: List[str] = []

    if report.get("disease_class") not in ALLOWED_DISEASE:
        problems.append("Missing/invalid disease_class.")

    lt = report.get("layers_table")
    if not isinstance(lt, list) or len(lt) != 7:
        problems.append("layers_table must be a list of exactly 7 items in the required order.")
    else:
        names = [row.get("layer") for row in lt]
        if names != ALLOWED_LAYERS:
            problems.append(f"layers_table must be exactly {ALLOWED_LAYERS} in order.")
        for i, row in enumerate(lt):
            if row.get("status") not in ALLOWED_STATUS:
                problems.append(f"layers_table[{i}].status must be one of {ALLOWED_STATUS}.")

    bt = report.get("biomarker_table")
    if bt is None or not isinstance(bt, list):
        problems.append("Missing/invalid biomarker_table.")
    else:
        if len(bt) > 4:
            problems.append("biomarker_table must have 4 or fewer items.")
        for i, item in enumerate(bt):
            if not isinstance(item, dict):
                problems.append(f"biomarker_table[{i}] must be an object.")
                continue
            for k in ("name", "location", "relevance"):
                if k not in item or not isinstance(item[k], str) or not item[k].strip():
                    problems.append(f"biomarker_table[{i}] missing/invalid '{k}'.")

    cs = report.get("clinical_summary")
    if not isinstance(cs, str) or not cs.strip():
        problems.append("Missing/invalid clinical_summary.")

    rl = report.get("roi_list")
    if rl is None or not isinstance(rl, list):
        problems.append("Missing/invalid roi_list.")
    else:
        if len(rl) > 8:
            problems.append("roi_list must have 8 or fewer items.")
        for i, r in enumerate(rl):
            if not isinstance(r, dict):
                problems.append(f"roi_list[{i}] must be an object.")
                continue

            biomarker_ref = r.get("biomarker_ref")
            if not isinstance(biomarker_ref, int) or biomarker_ref < 0:
                problems.append(f"roi_list[{i}].biomarker_ref must be a non-negative integer.")

            if r.get("type") not in ALLOWED_ROI_TYPES:
                problems.append(f"roi_list[{i}].type must be one of {ALLOWED_ROI_TYPES}.")

            if r.get("target") not in ALLOWED_TARGETS:
                problems.append(f"roi_list[{i}].target must be one of {ALLOWED_TARGETS}.")

            conf = r.get("confidence")
            if not isinstance(conf, (int, float)) or not _in_01(conf):
                problems.append(f"roi_list[{i}].confidence must be in [0,1].")

            has_coords = isinstance(r.get("coords"), list)
            has_points = isinstance(r.get("points"), list)

            if has_coords and has_points:
                problems.append(f"roi_list[{i}] must not have both coords and points.")
            if not has_coords and not has_points:
                problems.append(f"roi_list[{i}] must have either coords (box) or points (poly).")

            if has_coords:
                c = r["coords"]
                if len(c) != 4 or not all(_in_01(v) for v in c):
                    problems.append(f"roi_list[{i}].coords must be [x,y,w,h] with values in [0,1].")

            if has_points:
                pts = r["points"]
                ok = isinstance(pts, list) and len(pts) >= 3
                if ok:
                    for p in pts:
                        if (
                            not isinstance(p, list)
                            or len(p) != 2
                            or not _in_01(p[0])
                            or not _in_01(p[1])
                        ):
                            ok = False
                            break
                if not ok:
                    problems.append(
                        f"roi_list[{i}].points must be [[x,y],...] with x,y in [0,1] and at least 3 points."
                    )

            if isinstance(bt, list) and isinstance(biomarker_ref, int):
                if not (0 <= biomarker_ref < len(bt)):
                    problems.append(f"roi_list[{i}].biomarker_ref out of range (0..{len(bt)-1}).")

    if problems:
        print("Validation warnings:")
        for p in problems:
            print("-", p)


def generate_oct_report(
    disease_class: str,
    image_path: Optional[str],
    segmentation_path: Optional[str],
    model: str = "gpt-5",
    base_output_tokens: int = 2000,
) -> Optional[Dict[str, Any]]:
    """
    Generate a structured OCT report with optional ROIs.

    Args:
      disease_class: One of ALLOWED_DISEASE (echoed exactly in output).
      image_path: Path to raw OCT B-scan (optional).
      segmentation_path: Path to segmentation image (optional).
      model: OpenAI model name.
      base_output_tokens: Starting token budget, escalated on truncation.

    Returns:
      Parsed report dict if successful, otherwise None.
    """
    if disease_class not in ALLOWED_DISEASE:
        raise ValueError(f"disease_class must be one of {ALLOWED_DISEASE}")

    client = get_client()

    prompt = build_prompt(disease_class)
    tools = build_tools_schema(disease_class)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    if image_path:
        mime1 = guess_data_mime(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime1};base64,{encode_image_to_base64(image_path)}"},
            }
        )

    if segmentation_path:
        mime2 = guess_data_mime(segmentation_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime2};base64,{encode_image_to_base64(segmentation_path)}"},
            }
        )
        content.append(
            {
                "type": "text",
                "text": (
                    "Use the segmentation's vertical order to map layers from top (NFL) to bottom (RPE) "
                    "and set each layer's status. If a boundary is ambiguous, do not infer; default to 'normal' "
                    "unless a clear abnormality is visible."
                ),
            }
        )

    # Token budgets increase on truncation (length finish_reason).
    budgets = [base_output_tokens * 2, base_output_tokens * 3]

    for attempt, budget in enumerate(budgets, start=1):
        print(f"[attempt {attempt}] Requesting tool call with max_completion_tokens={budget}")

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a retina-specialized medical VLM. "
                            "Call the function immediately; do not output prose."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "submit_oct_report"}},
                max_completion_tokens=budget,
            )

        except TypeError as e:
            # Backward compatibility for older SDKs that do not support the tools parameter.
            print("[compat] tools not supported by this SDK; retrying with legacy functions:", e)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a retina-specialized medical VLM. "
                                "Return ONLY via the named function; no prose."
                            ),
                        },
                        {"role": "user", "content": content},
                    ],
                    functions=[tools[0]["function"]],
                    function_call={"name": "submit_oct_report"},
                    max_completion_tokens=budget,
                )
            except Exception as e2:
                print("[compat] Legacy functions call failed:", e2)
                continue

        except Exception as e:
            print(f"[attempt {attempt}] chat.completions failed:", e)
            continue

        report = _extract_structured_json(resp)
        if report:
            # Enforce exact echo of the input disease_class.
            if report.get("disease_class") != disease_class:
                print("[warn] Output disease_class deviated; overwriting with input value.")
                report["disease_class"] = disease_class

            validate_and_warn(report)

            print("\nStructured OCT Report:")
            print(json.dumps(report, indent=2))
            return report

        # Fallback: attempt to parse plain message content as JSON.
        ch = resp.choices[0]
        text = (getattr(ch.message, "content", "") or "").strip()
        if text:
            try:
                report = json.loads(text)
                if report.get("disease_class") != disease_class:
                    report["disease_class"] = disease_class
                validate_and_warn(report)

                print("\nStructured OCT Report (parsed from text):")
                print(json.dumps(report, indent=2))
                return report
            except Exception:
                pass

        finish_reason = getattr(ch, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        print(f"[attempt {attempt}] No tool args or parsable JSON. finish_reason={finish_reason}; usage={usage}")

        if finish_reason == "length":
            print("[info] Hit length limit; increasing token budget and retrying.")

    print("Failed to obtain a valid report after all attempts.")
    return None


if __name__ == "__main__":
    # disease_class must be one of: "DME", "Drusen", "CNV", "Normal"
    disease_class = "CNV"

    # Optional input images; script will skip any that do not exist.
    image_path = "CNV2.png"
    segmentation_path = "segmentation.png"

    report = generate_oct_report(
        disease_class=disease_class,
        image_path=image_path if os.path.exists(image_path) else None,
        segmentation_path=segmentation_path if os.path.exists(segmentation_path) else None,
        model="gpt-5",
        base_output_tokens=2000,
    )

    if report:
        out_path = "oct_structured_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to '{out_path}'")
