# Hopper-medium-v2 Exp

问题就是在 cql loss suppress 了 random a，提升了 in-sample 的 a。

只用 fake data 算 Q，用 real-data 来算 CQL loss。==> 90
(COMBOAgent_cqlreal_s0_20220404_060427.log)

不用 CQL loss。==> 20-90 波动很大。
(COMBOAgent_nocql_s0_20220404_060708)
