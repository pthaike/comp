create database bird;

mysql --local-infile=1 -u root -p

-- 1:230355
create table item_feature(
	date varchar(10),
	item_id varchar(15),
	cate_id varchar(10),
	cate_level_id varchar(10),
	brand_id varchar(10),
	supplier_id varchar(10),
	pv_ipv int(32),
	pv_uv int(32),
	cart_ipv int(32),
	cart_uv int(32),
	collect_uv int(32),
	num_gmv int(32),
	amt_gmv int(32),
	qty_gmv int(32),
	unum_gmv int(32),
	amt_alipay int(32),
	num_alipay int(32),
	qty_alipay int(32),
	unum_alipay int(32),
	ztc_pv_ipv int(32),
	tbk_pv_ipv int(32),
	ss_pv_ipv int(32),
	jhs_pv_ipv int(32),
	ztc_pv_uv int(32),
	tbk_pv_uv int(32),
	ss_pv_uv int(32),
	jhs_pv_uv int(32),
	num_alipay_njhs int(32),
	amt_alipay_njhs int(32),
	qty_alipay_njhs int(32),
	unum_alipay_njhs int(32)
);

-- 1: 950120
create table item_store_feature(
	date varchar(10),
	item_id varchar(15),
	store_code varchar(10),
	cate_id varchar(10),
	cate_level_id varchar(10),
	brand_id varchar(10),
	supplier_id varchar(10),
	pv_ipv int(32),
	pv_uv int(32),
	cart_ipv int(32),
	cart_uv int(32),
	collect_uv int(32),
	num_gmv int(32),
	amt_gmv int(32),
	qty_gmv int(32),
	unum_gmv int(32),
	amt_alipay int(32),
	num_alipay int(32),
	qty_alipay int(32),
	unum_alipay int(32),
	ztc_pv_ipv int(32),
	tbk_pv_ipv int(32),
	ss_pv_ipv int(32),
	jhs_pv_ipv int(32),
	ztc_pv_uv int(32),
	tbk_pv_uv int(32),
	ss_pv_uv int(32),
	jhs_pv_uv int(32),
	num_alipay_njhs int(32),
	amt_alipay_njhs int(32),
	qty_alipay_njhs int(32),
	unum_alipay_njhs int(32)
);

--1: 6000
create table config(
	item_id varchar(15),
	store_code varchar(10),
	a_b varchar(30)
);

--1:
-- create table target(
-- 	item_id varchar(15),
-- 	store_code varchar(10),
-- 	target
-- );

LOAD DATA LOCAL INFILE 'item_feature1.csv' INTO TABLE item_feature FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';
LOAD DATA LOCAL INFILE 'item_store_feature1.csv' INTO TABLE item_store_feature FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';
LOAD DATA LOCAL INFILE 'config1.csv' INTO TABLE config FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';

select date, count(*) from item_store_feature where item_id = '84923' group by date;

select date, qty_alipay from item_store_feature where item_id = '84923' group by date;


select * from config where item_id = '9595';

-- 1000 products
select distinct item_id from config;


select item_id,count(*) from item_feature group by item_id order by count(*) limit 100;

select item_id,count(*) from item_feature where date > '20151215' and qty_alipay > 0 and num_alipay_njhs > 0 group by item_id order by count(*);

select date, qty_alipay from item_feature where item_id = '79276';



-- view
 create view item_buy as select date, item_id, qty_alipay_njhs from item_feature;
 create view item_store_buy as select date, item_id, store_code, qty_alipay_njhs from item_store_feature;

 select * from item_store_buy where item_id = '139254';

 select distinct item_id from config;

 select distinct item_id from config into outfile '/tmp/item.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 


 select date, item_id, pv_ipv from item_feature; 

