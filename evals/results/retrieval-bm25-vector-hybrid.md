## Retrieval eval (27 queries, 30 passages)

| Retriever | recall@3 | hit@1 | MRR | recall@3 brand | recall@3 issue | recall@3 keyword | recall@3 paraphrase | p50 latency |
|---|---|---|---|---|---|---|---|---|
| bm25 | 83% | 74% | 0.79 | 83% | 96% | 100% | 57% | 0.0 ms |
| bm25+expand | 87% | 85% | 0.88 | 100% | 96% | 100% | 57% | 0.0 ms |
| vector | 96% | 82% | 0.90 | 100% | 92% | 100% | 100% | 212.9 ms |
| vector+expand | 96% | 93% | 0.96 | 100% | 92% | 100% | 100% | 257.9 ms |
| hybrid | 91% | 78% | 0.87 | 100% | 96% | 100% | 71% | 243.0 ms |
| hybrid+expand | 91% | 89% | 0.93 | 100% | 96% | 100% | 71% | 273.5 ms |

### Misses (relevant passage not in the top 3)

- `bm25` r11 "strength_mg was not found for atorvastatin": wanted ['G16', 'G30'], got ['G16', 'G17', 'G22']
- `bm25` r13 "patient on a blood thinner needs something for pain": wanted ['G21'], got ['G23', 'G10', 'G30']
- `bm25` r14 "can I give the diabetes tablet as a shot": wanted ['G08'], got ['G24', 'G13', 'G06']
- `bm25` r15 "older person taking painkillers for arthritis, worried about stomach bleeding": wanted ['G04'], got ['G03', 'G20', 'G14']
- `bm25` r27 "Coumadin together with Brufen": wanted ['G21'], got []
- `bm25+expand` r11 "strength_mg was not found for atorvastatin": wanted ['G16', 'G30'], got ['G16', 'G17', 'G22']
- `bm25+expand` r13 "patient on a blood thinner needs something for pain": wanted ['G21'], got ['G23', 'G10', 'G30']
- `bm25+expand` r14 "can I give the diabetes tablet as a shot": wanted ['G08'], got ['G24', 'G13', 'G06']
- `bm25+expand` r15 "older person taking painkillers for arthritis, worried about stomach bleeding": wanted ['G04'], got ['G03', 'G20', 'G14']
- `vector` r08 "Paediatric dosing needs pharmacist review": wanted ['G28', 'G06'], got ['G28', 'G16', 'G17']
- `vector` r11 "strength_mg was not found for atorvastatin": wanted ['G16', 'G30'], got ['G16', 'G14', 'G01']
- `vector+expand` r08 "Paediatric dosing needs pharmacist review": wanted ['G28', 'G06'], got ['G28', 'G16', 'G17']
- `vector+expand` r11 "strength_mg was not found for atorvastatin": wanted ['G16', 'G30'], got ['G16', 'G14', 'G01']
- `hybrid` r11 "strength_mg was not found for atorvastatin": wanted ['G16', 'G30'], got ['G16', 'G22', 'G17']
- `hybrid` r13 "patient on a blood thinner needs something for pain": wanted ['G21'], got ['G23', 'G10', 'G20']
- `hybrid` r14 "can I give the diabetes tablet as a shot": wanted ['G08'], got ['G13', 'G01', 'G24']
- `hybrid+expand` r11 "strength_mg was not found for atorvastatin": wanted ['G16', 'G30'], got ['G16', 'G22', 'G17']
- `hybrid+expand` r13 "patient on a blood thinner needs something for pain": wanted ['G21'], got ['G23', 'G10', 'G20']
- `hybrid+expand` r14 "can I give the diabetes tablet as a shot": wanted ['G08'], got ['G13', 'G01', 'G24']
