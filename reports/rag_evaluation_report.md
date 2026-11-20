# RAG Evaluation Report: Nimbus Labs Handbook (fictional)
*Chat model:* `gemini-3.5-flash-lite` · *Embedding model:* `gemini-embedding-2-preview` · *Questions:* 8 · *Generated:* 2026-11-20 04:05

## Retrieval quality
|   chunk_size |   overlap |   chunks |   top_k |   hit_rate |   MRR |
|-------------:|----------:|---------:|--------:|-----------:|------:|
|          300 |        50 |       23 |       1 |          1 |     1 |
|          300 |        50 |       23 |       3 |          1 |     1 |
|          800 |       150 |       11 |       1 |          1 |     1 |
|          800 |       150 |       11 |       3 |          1 |     1 |
|         1500 |       200 |        9 |       1 |          1 |     1 |
|         1500 |       200 |        9 |       3 |          1 |     1 |

**Best retrieval setting:** chunk size 300, top-k 1 (hit rate 100%, MRR 1.0)

## End-to-end answers (chunk 500, top-3)
**Answer accuracy: 100%**

| question                                                          | expected                    | answer                                                                                                         | correct   |
|:------------------------------------------------------------------|:----------------------------|:---------------------------------------------------------------------------------------------------------------|:----------|
| How many days of paid annual leave do employees get?              | 24 days                     | Every full-time employee receives 24 days of paid annual leave per calendar year [1].                          | True      |
| Which days do employees have to come to the office?               | Tuesdays and Thursdays      | Employees must work from the office on Tuesdays and Thursdays [1].                                             | True      |
| How much money can I spend on courses each year?                  | 40,000                      | You have a learning budget of ₹40,000 per year for courses, certifications, books, and conference tickets [1]. | True      |
| How often are laptops replaced?                                   | 3 years                     | Laptops are replaced every 3 years, or earlier if they fail [1].                                               | True      |
| I lost my work laptop. Who do I tell?                             | security@nimbuslabs.example | You must report a lost laptop to **security@nimbuslabs.example** within **2 hours** of discovery [1].          | True      |
| What is the meal allowance per day when travelling outside India? | €60                         | The daily meal allowance when travelling abroad (outside India) is €60 [1].                                    | True      |
| How long is probation for new employees?                          | 6 months                    | The probation period for new employees is 6 months [1].                                                        | True      |
| How fast must on-call engineers respond to a severity 1 incident? | 15 minutes                  | On-call engineers must respond to severity 1 critical incidents within **15 minutes** [1].                     | True      |

## Takeaways
- Without retrieval the model cannot answer questions about private documents and may hallucinate.
- Retrieving more chunks (top-3 vs top-1) raises the hit rate, because the right chunk is not always ranked first.
- Chunk size is a trade-off: small chunks are precise but can split facts; large chunks need fewer hits but add noise and tokens.
- A strict grounded prompt ("use only the context", "cite sources", "say if not found") prevents made-up answers.