# Taxonomy extension run -- review log

- Parsed inputs/genus_statements.owl: 597 classes, 879 subclass_of edges, base_iri=http://example.org/genus-kg
- NOTE: expected backwards edge subclass_of(value, present_discounted_value) not found -- taxonomy may have changed since this fix was diagnosed; skipped
- NOTE: expected backwards edge subclass_of(value, present_value_of_the_cash_inflows_of_a_project) not found -- taxonomy may have changed since this fix was diagnosed; skipped
- Auto-detected glossary merge state of --owl-in: ALREADY MERGED
- Merged class 'economic_exposure' into 'economic_risk' directly in the graph (glossary was already merged upstream)
- Merged class 'transaction_exposure' into 'transaction_risk' directly in the graph (glossary was already merged upstream)
- Merged class 'translation_exposure' into 'translation_risk' directly in the graph (glossary was already merged upstream)
- No cycles detected after fixes + glossary merge -- taxonomy is a valid DAG.
- Wrote out\principles_of_finance_taxonomy_extended.owl
- Loaded 27 Adobe concepts; 150 candidate genus classes; 37 predicates in vocabulary
- Wrote out\adobe_taxonomy_extension.owl
- Wrote out\adobe_concept_taxonomy_mapping.csv
