Return JSON with a top-level "records" array.

Extract a record only when the source text directly concerns climate action. A
valid record must include explicit evidence for at least one of these SDG 13
areas:
- 13.1 climate resilience, climate adaptation, disaster risk reduction, extreme weather preparedness
- 13.2 climate policy, greenhouse gas mitigation, emissions reduction, carbon markets/pricing, NDCs, net zero, carbon neutrality, carbon peaking
- 13.3 climate education, climate awareness, climate capacity building, climate early warning
- 13.a climate finance commitments under UNFCCC/Paris Agreement
- 13.b climate planning/capacity for vulnerable countries or communities

Reject weakly related content. Return no record for generic digital economy,
telecommunications, AI, elderly services, Taiwan/mainland relations, general
science and technology, generic agriculture, general economic growth, routine
industrial policy, or generic sustainability disclosure unless the exact source
text explicitly links it to climate change, GHG emissions, carbon, mitigation,
adaptation, resilience, disaster risk, climate finance, climate education, NDC,
UNFCCC, or Paris Agreement.

For every emitted record:
- sdg_goal must be exactly "SDG 13".
- sdg_target must be exactly one of: "13.1", "13.2", "13.3", "13.a", "13.b".
- sdg_confidence must be a number from 0 to 1.
- policy_content_original must preserve the complete relevant policy clause or policy fragment.
- evidence_quote must be a brief verbatim quote used only to validate the extraction; keep it shorter than policy_content_original.
- If you cannot assign a valid SDG 13 target from evidence, do not emit the record.

Use null for unavailable optional fields. Do not use empty strings.
Return JSON only.
