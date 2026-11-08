**Table 1 - Thresholds**

| Threshold | Rule | Validation | Test |
|---|---|---|---|
| Relevance | held-out rating ≥ 4 | 55.1 % of ratings relevant; median 11 per evaluated user; 31 users have none and are not scored | 53.7 % of ratings relevant; median 18 per evaluated user; 19 users have none and are not scored |
| History | ≥ 1 earlier rating to be evaluated | 3 users (168 ratings) not scored | 3 users (547 ratings) not scored |
| Item filter | ≥ 5 ratings in the fitting data | 305 of 3,678 items dropped; 612 fitting ratings (0.07 %); 83 held-out ratings unreachable, 29 of them relevant | 300 of 3,688 items dropped; 612 fitting ratings (0.06 %); 192 held-out ratings unreachable, 105 of them relevant |
| Candidate set | items kept by the filter | 3,373 items | 3,388 items |

**Table 2 - Split**

| Set | Period (UTC) | Ratings | Users | Users with no earlier ratings | Evaluated users |
|---|---|---|---|---|---|
| Train | 25 Apr – 31 Dec 2000 | 904,757 (90.5 %) | 6,034 | – | – |
| Validation | 1 Jan – 30 Jun 2001 | 47,314 (4.7 %) | 926 | 3 | 892 |
| Test | 1 Jul 2001 – 28 Feb 2003 | 48,138 (4.8 %) | 799 | 3 | 777 |
