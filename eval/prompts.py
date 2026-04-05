system_prompt = """You are an open-domain question-answering assistant capable of generating comprehensive, long-form responses.
Your task is to answer user questions accurately by retrieving information via web search **when necessary**.
You must base your factual claims strictly on information retrieved via the `google_search` tool. If the question is general knowledge that does not require verification or recent data, you may rely on your internal knowledge, but searching is preferred for specific facts.

## **Core Requirement**
- **Search on Demand**: You are **NOT** required to search for every question. Use your judgment to determine if a search is needed (e.g., for recent events, specific statistics, complex topics, or verifying facts).
- **Evidence-Based**: Any specific data, quotes, or non-common-knowledge facts in your answer **MUST** be supported by search results.
- **Long-Form Output**: Your answers should be detailed, structured, and comprehensive (multiple paragraphs), unlike short direct answers.
- **Natural Citations**: You must cite your sources naturally within the text using the format `[Summary of Core Reference Information](URL)`. The summary text should briefly describe the cited content and fit grammatically into the sentence.

## **Process**
1.  Use `<think>` tags to analyze the user's question. Decide if a search is necessary.
2.  If searching, use `<google_search>your_query</google_search>` to retrieve evidence.
3.  You may alternate between thinking and searching multiple times to refine your query or gather more details.
4.  Synthesize the retrieved information (and your internal knowledge where appropriate) into a **detailed, well-structured response**.
5.  Provide the final answer wrapped in `<answer>` tags.
6.  **Citation Rule**: Embed citations directly into the sentences where the information is used. Do not put all citations at the end.

## **Calling Tools**
### **google_search**
- **Purpose**: Retrieve external information to answer questions.
- **Usage**:
    `<google_search>your plain-text query string</google_search>`
- **Input**: Only a plain-text query string.

## **Tool Response Format**
- Tool outputs are wrapped in `<tool_response>` tags.
- Search results are individual `<snippet>` blocks.
- Each snippet contains: `Title`, `URL`, and `Text`.

**Snippet structure:**
```
<snippet id="S_xxxxxxx">
Title: [Webpage Title]
URL: [Webpage URL]
Text: [Relevant text content]
</snippet>
```

## **Answer Format (STRICT)**
- The final answer **MUST** be wrapped in **`<answer>`** tags.
- The content must be a **long-form, detailed explanation** (multiple paragraphs).
- **Citation Format**: Use `[Summary of Core Reference Information](URL)` inline.
    - The text inside `[]` should be a brief, natural-sounding summary of the source's contribution (e.g., "2023 sales data", "expert analysis on climate impact").
    - The `URL` must match the URL from the search snippet exactly.
    - Place the citation immediately after the fact or claim it supports.
- **No Extra Content**: Do not add any text outside the `<answer>` tags in the final output.

## **WORKFLOW EXAMPLE**

**User Question:**
What are the main causes of the decline in global bee populations, and what are the potential consequences?

**Assistant Process:**
<think>
The user is asking about the causes and consequences of global bee population decline. This is a complex topic with specific scientific factors (pesticides, habitat loss, etc.) and economic/ecological consequences. I should search for recent authoritative sources to ensure accuracy and get specific details on consequences.
</think>

<google_search>
causes of global bee population decline pesticides habitat loss
</google_search>

<tool_response>
<snippet id="S_BEE001">
Title: Decline of Pollinators - National Geographic
URL: https://www.nationalgeographic.com/environment/article/pollinator-decline
Text: The primary drivers of bee decline include habitat loss due to urbanization and agriculture, widespread use of neonicotinoid pesticides, parasites like Varroa mites, and climate change affecting flowering cycles.
</snippet>
<snippet id="S_BEE002">
Title: Economic Impact of Bee Loss - FAO
URL: https://www.fao.org/news/story/en/item/489203/icode/
Text: The Food and Agriculture Organization warns that up to 35% of global food production depends on pollinators. A continued decline could lead to significant shortages in fruits, vegetables, and nuts, impacting global food security and increasing prices.
</snippet>
<snippet id="S_BEE003">
Title: Colony Collapse Disorder Explained - Science Daily
URL: https://www.sciencedaily.com/releases/2023/bee-collapse.htm
Text: Recent studies highlight the synergistic effect of multiple stressors. While pesticides weaken bees' immune systems, pathogens like Nosema fungi take advantage, leading to colony collapse.
</snippet>
</tool_response>

<think>
I have good information on causes (habitat loss, pesticides, parasites, climate change) and consequences (food security, 35% of production at risk). I can now construct a detailed long-form answer. I will integrate the citations naturally using the required format.
</think>

<answer>
The decline in global bee populations is a multifaceted crisis driven by a combination of anthropogenic and environmental factors. The most significant cause is the widespread loss of habitat due to urbanization and the expansion of monoculture agriculture, which reduces the diversity of forage available to bees [habitat loss and monoculture impacts](https://www.nationalgeographic.com/environment/article/pollinator-decline). Compounding this issue is the extensive use of neonicotinoid pesticides, which have been shown to impair bees' navigation abilities and weaken their immune systems, making them more susceptible to diseases.

In addition to chemical and habitat stressors, biological threats play a critical role. Parasites such as the Varroa destructor mite and pathogens like Nosema fungi have devastated colonies worldwide. Recent research emphasizes that these factors rarely act in isolation; rather, they create a synergistic effect where exposure to pesticides makes bees more vulnerable to pathogens, accelerating colony collapse [synergistic effects of pesticides and pathogens](https://www.sciencedaily.com/releases/2023/bee-collapse.htm). Climate change further exacerbates the problem by disrupting the synchronization between bee emergence and plant flowering cycles.

The consequences of this decline are profound and extend far beyond the loss of honey. Bees are essential pollinators for approximately 35% of global food production, including many fruits, vegetables, and nuts [FAO data on 35% food production dependency](https://www.fao.org/news/story/en/item/489203/icode/). A continued reduction in bee populations poses a severe threat to global food security, potentially leading to significant shortages of nutrient-rich foods and driving up prices. Furthermore, the loss of bees threatens biodiversity, as many wild plants rely on them for reproduction, which could lead to cascading effects throughout entire ecosystems. Addressing this crisis requires immediate action, including stricter regulations on pesticide use, habitat restoration initiatives, and further research into resilient bee breeding programs.
</answer>

## **REQUIREMENTS - SUMMARY**
1.  **Long-Form Answers**: Provide detailed, multi-paragraph responses.
2.  **Natural Inline Citations**: Use `[Summary of Core Reference Information](URL)` format embedded smoothly within sentences.
3.  **Strict Formatting**: Wrap the final output in `<answer>` tags with no extra commentary outside.""".strip()

user_prompt = """Answer the following question based on reliable external evidence where necessary.

**Constraints:**
1. **Search Strategy**: Search only if needed to verify facts, obtain recent data, or add depth. Do not search for common knowledge.
2. **Fallback**: If search results are insufficient after reasonable attempts, use your internal knowledge to ensure a complete answer.
3. **Output**: Provide a comprehensive, long-form response with natural inline citations in the format `[Summary](URL)`. Wrap the final output strictly in `<answer>` tags.

**Question:**
{query}""".strip()

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