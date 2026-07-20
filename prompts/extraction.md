Return JSON with a top-level "records" array. Each record must correspond to one
policy paragraph, clause, or independently citable policy fragment from the input.

For each record, keep only the supplied output schema fields. Preserve original
policy wording in policy_content_original, optionally add a concise
policy_content_summary, assign sdg_goal and sdg_target only when supported by the
source text, and include a short evidence_quote copied from the source.

Do not include markdown. Return JSON only.
