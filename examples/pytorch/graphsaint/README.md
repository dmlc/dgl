## GraphSAINT

> This README is just a brief description of codes instead of a formal README. The original one is renamed as _README.md



## Codes

- sampler.py: the sampler codes of the original author @lt610
- sampler_jiahanli.py: the refactored version of sampler
- _train_sampling: the training codes of the original author @lt610
- train_sampling: the refactored version of training codes
- other files (code) are basically the same as the original version



## Results

- The results are measured by the same way of the original version of @lt610. Next step I'd employ line-profiler to measure more accurate time consumption.

| Task: Node   | PPI         | Flickr      | Reddit      | Yelp        | Amazon      |       |       |
| ---------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----- | ----- |
| **F1-micro**           |             |             |             |             |             |       |       |
| Paper                  | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 | 0.641±0.000 | 0.782±0.004 |       |       |
| Running                | 0.9628      | 0.5077      | 0.9622      | 0.6393      | 0.7695      |       |       |
| DGL                    | 0.9618      | 0.4828      | 0.9621      | 0.6360      | 0.7748      |       |       |
| **New**                | **0.9666**  | **0.4747**  | **0.9593**  | **0.6326**  |             |       |       |
| **Time(s)**            |             |             |             |             |             |       |       |
| Sampling(Running)      | 0.77        | 0.65        | 7.46        | 26.29       | 571.42      |       |       |
| Normalization(Running) | 0.69        | 2.84        | 11.54       | 32.72       | 407.20      |       |       |
| **sum**     | **1.46**    | **3.49**    | **19**      | **59.01**   | **978.62**  | **0** | **0** |
|                        |             |             |             |             |             |       |       |
| Normalization(DGL)     | 1.04        | 0.41        | 21.05       | 68.63       | 2006.94     |       |       |
| Sampling(DGL)          | 0.24        | 0.57        | 5.06        | 30.04       | 163.75      |       |       |
| **sum**           | **1.28**    | **0.98**    | **26.11**   | **98.67**   | **2170.69** | **0** | **0** |
|                        |             |             |             |             |             |       |       |
| **New**                | **2.32(0)** | **2.14(4)** | **19.63**   | **54.04**   | **926.76**  |       |       |
|                        |             |             |             |||||

| Task: Edge             | PPI         | Flickr      | Reddit      | Yelp         | Amazon      |       |       |
| ---------------------- | ----------- | ----------- | ----------- | ------------ | ----------- | ----- | ----- |
| **F1-micro**           |             |             |             |              |             |       |       |
| Paper                  | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 | 0.653±0.003  | 0.807±0.001 |       |       |
| Running                | 0.9810      | 0.5066      | 0.9656      | 0.6531       | 0.8071      |       |       |
| DGL                    | 0.9818      | 0.5054      | 0.9653      | 0.6517       | exceed      |       |       |
| **New**                | **0.9814**  | **0.5035**  | **0.9653**  | **0.6527**   |             |       |       |
| **Time(s)**            |             |             |             |              |             |       |       |
| Sampling(Running)      | 0.72        | 0.56        | 4.46        | 12.38        | 101.76      |       |       |
| Normalization(Running) | 0.68        | 2.62        | 9.42        | 26.64        | 62.59       |       |       |
| **sum**                | **1.4**     | **3.18**    | **13.88**   | **39.02**    | **164.35**  | **0** | **0** |
|                        |             |             |             |              |             |       |       |
| Sampling(DGL)          | 0.50        | 0.72        | 53.88       | 254.63       | exceed      |       |       |
| Normalization(DGL)     | 0.61        | 0.38        | 14.69       | 23.63        | exceed      |       |       |
| **sum**                | **1.11**    | **1.1**     | **68.57**   | **278.26**   | **0**       | **0** | **0** |
|                        |             |             |             |              |             |       |       |
| **New**                | **2.90**    | **4.44**    | **31.46**   | **53.86(4)** |             |       |       |
|                        |             |             |             |              |             |       |       |

| Task: Random Walk      | PPI         | Flickr      | Reddit      | Yelp        | Amazon      |       |       |
| ---------------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----- | ----- |
| **F1-micro**           |             |             |             |             |             |       |       |
| Paper                  | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 | 0.653±0.003 | 0.815±0.001 |       |       |
| Running                | 0.9812      | 0.5104      | 0.9648      | 0.6527      | 0.8131      |       |       |
| DGL                    | 0.9818      | 0.5018      | 0.9649      | 0.6516      | 0.8150      |       |       |
| **New**                | **0.9823**  | **0.4817**  | **0.9640**  | **0.6502**  |             |       |       |
| **Time(s)**            |             |             |             |             |             |       |       |
| Sampling(Running)      | 0.83        | 1.22        | 6.69        | 18.84       | 209.83      |       |       |
| Normalization(Running) | 0.87        | 2.60        | 10.28       | 24.41       | 145.85      |       |       |
| **sum**                | **1.7**     | **3.82**    | **16.97**   | **43.25**   | **355.68**  | **0** | **0** |
|                        |             |             |             |             |             |       |       |
| Sampling(DGL)          | 0.28        | 0.63        | 4.02        | 22.01       | 55.09       |       |       |
| Normalization(DGL)     | 0.70        | 0.42        | 18.34       | 32.16       | 683.96      |       |       |
| **sum**                | **0.98**    | **1.05**    | **22.36**   | **54.17**   | **739.05**  | **0** | **0** |
|                        |             |             |             |             |             |       |       |
| **New**                | **3.37**    | **3.72**    | **19.04**   | **19.86**   | **117.52**  |       |       |
|                        |             |             |             |             |             |       |       |