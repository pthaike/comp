DROP TABLE IF EXISTS item_feature;
CREATE TABLE item_feature AS
SELECT * FROM tianchi_data.item_store_feature;


DROP TABLE IF EXISTS item_store_feature;
CREATE TABLE item_store_feature AS
SELECT * FROM tianchi_data.item_store_feature;

DROP TABLE IF EXISTS config;
CREATE TABLE config AS
SELECT * FROM tianchi_data.config;


--FEATURE

DROP TABLE IF EXISTS feature;
CREATE TABLE feature AS
SELECT qty_alipay_njhs, pv_ipv, pv_uv, cart_ipv, cart_uv, collect_uv, num_gmv, qty_gmv, unum_gmv, num_alipay, qty_alipay, ztc_pv_ipv, tbk_pv_ipv, ss_pv_ipv, jhs_pv_uv, num_alipay_njhs, unum_alipay_njhs FROM
item_feature 


SELECT item_id, store_code, sum(unum_alipay_njhs) AS target FROM item_store_feature WHERE thedate >= '20151217' AND thedate <= '20151227' GROUP BY item_id;