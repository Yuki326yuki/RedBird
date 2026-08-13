Return a JSON object with a top-level "records" array.

Extract only explicit policy measures that directly concern climate change, greenhouse-gas mitigation, climate adaptation, resilience, climate hazards, climate education, climate finance, NDCs, UNFCCC, or the Paris Agreement.

For each record return only:
- policy_content_original: a complete verbatim policy clause or compact passage from the source.
- policy_content_summary: a faithful one-sentence summary with no inference.
- evidence_quote: the shortest verbatim phrase that proves the climate connection.

Do not assign sdg_goal or sdg_target in this stage. Reject generic green development, general energy, agriculture, technology, finance, disclosure, or economic policy unless the source explicitly connects the measure to climate action. If there is no direct policy measure, return {"records": []}.

Source documents may be in any language represented in the manifest. Preserve every verbatim quote and original policy passage in the source language.
