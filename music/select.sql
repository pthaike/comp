#查询每个歌手每天的播放次数
select songs.artist_id, actions.ds, count(*) from actions, songs where songs.song_id = actions.song_id group by songs.artist_id, actions.ds;

#利用视图来查询每个歌手每天的播放次数
select * from playtimes;

# all plays every day
select actions.ds, count(*) from actions group by actions.ds;

select actions.ds, count(*) from actions where actions.action_type = 0 group by actions.ds;

select actions.ds, count(*) from actions where actions.action_type = 1 group by actions.ds;

select actions.ds, count(*) from actions where actions.action_type = 2 group by actions.ds;

#查询特定用户的playtimes记录

select * from playtimes where artist_id = '023406156015ef87f99521f3b343f71f';
select * from playtimes where artist_id = '023406156015ef87f99521f3b343f71f' into outfile '/tmp/023406156015ef87f99521f3b343f71f.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 

select * from playtimes where artist_id = '2e14d32266ee6b4678595f8f50c369ac' into outfile '/tmp/2e14d32266ee6b4678595f8f50c369ac.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 
select * from playtimes where artist_id = '25739ad1c56a511fcac86018ac4e49bb' into outfile '/tmp/25739ad1c56a511fcac86018ac4e49bb.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 
select * from playtimes where artist_id = '445a257964b9689f115a69e8cc5dcb75' into outfile '/tmp/445a257964b9689f115a69e8cc5dcb75.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 

select * from playtimes where artist_id = '5e2ef5473cbbdb335f6d51dc57845437' into outfile '/tmp/5e2ef5473cbbdb335f6d51dc57845437.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 