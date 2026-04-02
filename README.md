# TTT-RAG


## Evaluation Pipeline Prompt Example Template

### 1. Extract Key Points

**Prompt**

Based on the question and answer, summarize ten key points that you consider to be the most crucial from the standard answer. Return the response in the following format: `{1.2.3....}`

Here is the question: `{question}`  
Here is the answer: `{answer}`  

Please do not provide any additional information.

**Output Example (Key Points)**

1. Multifocal electroretinogram (ERG) showed reduced signal in the right eye throughout the macula, confirming the diagnosis of AZOOR.  
2. Acute zonal occult outer retinopathy (AZOOR) was first described by Gass in 1993...  

---

### 2. Extract Diagnostic Reasoning

**Prompt**

Based on the question and answer, please provide a detailed summary of the diagnostic reasoning from the standard answer. Return the response in the following format: `{1.2.3....}`

Here is the question: `{question}`  
Here is the answer: `{answer}`  

Please do not provide any additional information.

**Output Example (Diagnostic Reasoning)**

1. The patient is a 7-year-old boy with a slowly growing, asymptomatic lump on the left lower neck since birth.  
2. Physical examination showed a yellowish, hump-like mass with a hairy surface and cartilage-like consistency near the left sternocleidomastoid muscle...  

---

### 3. Extract Evidence

**Prompt**

Based on the question and answer, please provide a detailed evidence list which is proposed by correct answer. Return the response in the following format: `{1.2.3....}`

Here is the question: `{question}`  
Here is the answer: `{answer}`  

Please do not provide any additional information.

**Output Example (Evidence)**

1. Slowly growing, asymptomatic lump on left lower neck since birth.  
2. Physical examination revealed a yellowish, hump-like mass with hairy surface and cartilage-like consistency.  
3. Ultrasonography indicated a hypoechoic, avascular, bulging nodule with an anechoic tubular structure.  
4. MRI demonstrated a protuberant nodule with diffuse...  

---

### 4. Key Points Score

**Prompt**

Act as a USMLE evaluator, your role involves assessing and comparing a medical student's explanation to the provided target answer. Begin the assessment by carefully reviewing the provided target answer. Then, based on the following specific criteria, determine the score for the student's answer.

Please judge whether the medical student's answer includes these key points (or some other relevant points, but the amount of points must be complete). For example, if the ground truth has 10 key points, and the student answer includes one key point, they will get 0.5 point (if the answer includes 5 points, the score should be 2.5).

Medical student's answer:  
`{answer}`

Key Points:  
`{Key Point}`

Please only return a float number (from 0 to 5). You should check each point one by one (do not judge based on language style such as fluency and so on. Only judge based on whether the student's answer includes correct, relevant, and complete key points). Do not generate any other information.
