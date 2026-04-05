claim_request_prompt_template = """\
Below you will receive a piece of text. Your task is:

1. Determine whether the text contains verifiable objective claims.
2. If verifiable objective claims exist in the text, you must extract these claims from the answer (regardless of whether these claims are true).
3. If the text does not contain any verifiable objective claims, return "no verifiable objective claims".

Response format:

* Claim 1
* Claim 2
...
(or "no verifiable objective claims")

The claims you extract must adhere to the following 3 principles:

1. Objectively verifiable: The claim must describe an objectively verifiable fact, not a subjective judgment, evaluation, or opinion.
2. Indivisible: The objective fact described by the claim cannot be further broken down.
3. Explicit meaning: Each claim must be a complete, self-contained sentence with all coreferences resolved. There should be no nouns or pronouns with unclear meaning.
 
Please strictly follow the above rules to complete the following task:
[Text]: {response}
[Verifiable objective claims]:""".strip()


claim_check_prompt_template="""You will determine if the following claim is true based on your knowledge. Answer only with "True" or "False".

Example:
Claim: The capital of France is London.
Answer: False

Example:
Claim: The capital of France is Paris.
Answer: True

Now, please evaluate the following:
Claim: {claim}
Answer:""".strip()


citation_extraction_template = """Your task is to analyze a research document and locate all inline citations that follow a specific markdown-style pattern.

The target citation format is: [descriptive text](hyperlink)
For instance: "Recent studies show that [global temperatures have risen by 1.2°C since pre-industrial times](https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature)."

Instructions for extraction:
- Scan the entire document and identify every citation matching the [text](url) pattern
- For each match, extract the factual claim being made along with its source URL
- Include sufficient surrounding context so the extracted fact is self-contained and verifiable
- When one statement references multiple sources (e.g., [claim A](url1) and [claim B](url2)), create separate entries for each
- Ignore any other reference styles such as numbered footnotes [1], parenthetical citations (Author, Year), or plain URLs
- Return an empty list if no valid [text](url) citations exist

Output format - a JSON array of objects:
[
    {{
        "fact": "The complete factual statement extracted from the document. Ensure proper escaping of quotes for JSON compatibility.",
        "url": "https://example.org/reference-page"
    }}
]

Document to analyze:
{report_text}

Respond with only the JSON array. Do not include any explanatory text or commentary.""".strip()


citation_judge_template = """You are a meticulous fact verification specialist. Your objective is to assess whether a given reference document substantiates a specific claim. Base your evaluation exclusively on the document content provided—do not incorporate prior knowledge or external information.

Rating criteria:

- "Fully supported": The document explicitly confirms all major aspects of the claim. Key facts, figures, and assertions are directly stated or unambiguously implied.
- "Partially supported": The document validates a substantial portion (over 50%) of the claim, but certain details remain unaddressed, unclear, or show minor discrepancies.
- "No support": The document fails to provide relevant evidence for the claim—either the content is unrelated, contradicts the assertion, or offers only superficial connections.

Inputs for evaluation:

Claim: {claim}

Document: {document}

Based strictly on the document above, determine the level of support. Output exactly one of these labels: Fully supported / Partially supported / No support""".strip()