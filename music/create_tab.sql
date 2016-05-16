#创建表
--------------------------------------------------------------

#歌曲数据，一共10,842条
create table songs(
	song_id varchar(50),
	artist_id varchar(50),
	publish_time varchar(16),
	song_init_plays int(32),
	language int(16),
	gender int(8),
	primary key (song_id, artist_id)
);

+-----------------+-------------+------+-----+---------+-------+
| Field           | Type        | Null | Key | Default | Extra |
+-----------------+-------------+------+-----+---------+-------+
| song_id         | varchar(50) | NO   | PRI |         |       |
| artist_id       | varchar(50) | NO   | PRI |         |       |
| publish_time    | varchar(16) | YES  |     | NULL    |       |
| song_init_plays | int(32)     | YES  |     | NULL    |       |
| language        | int(16)     | YES  |     | NULL    |       |
| gender          | int(8)      | YES  |     | NULL    |       |
+-----------------+-------------+------+-----+---------+-------+

#用户数据，一共5,652,232条
create table actions(
	user_id varchar(50),
	song_id varchar(50),
	gmt_create varchar(20),
	action_type int(8),
	ds varchar(16)
);

create table actions(
	user_id varchar(50),
	song_id varchar(50),
	gmt_create varchar(20),
	action_type int(8),
	ds date
);
+-------------+-------------+------+-----+---------+-------+
| Field       | Type        | Null | Key | Default | Extra |
+-------------+-------------+------+-----+---------+-------+
| user_id     | varchar(50) | YES  |     | NULL    |       |
| song_id     | varchar(50) | YES  |     | NULL    |       |
| gmt_create  | varchar(20) | YES  |     | NULL    |       |
| action_type | int(8)      | YES  |     | NULL    |       | 1，播放；2，下载，3，收藏
| ds          | varchar(16) | YES  |     | NULL    |       |
+-------------+-------------+------+-----+---------+-------+

========================================================
#导入csv数据
--------------------------------------------------------


#start the mysql with the option local infile 
mysql --local-infile=1 -u root -p

#load the song csv data into table songs
LOAD DATA LOCAL INFILE 'mars_tianchi_songs.csv' INTO TABLE songs FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';

#load the action csv data into table actions
LOAD DATA LOCAL INFILE 'mars_tianchi_user_actions.csv' INTO TABLE actions FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';


========================================================
#创建视图
--------------------------------------------------------

#create view that every artist's play times on every day, the record total 9146
create view playtimes as select songs.artist_id artist_id, count(*) as plays, actions.ds as ds from actions, songs where songs.song_id = actions.song_id group by songs.artist_id, actions.ds;

	+-----------+-------------+------+-----+---------+-------+
	| Field     | Type        | Null | Key | Default | Extra |
	+-----------+-------------+------+-----+---------+-------+
	| artist_id | varchar(50) | NO   |     |         |       |
	| plays     | bigint(21)  | NO   |     | 0       |       |
	| ds        | varchar(16) | YES  |     | NULL    |       |
	+-----------+-------------+------+-----+---------+-------+

# 9002
create view download as select songs.artist_id artist_id, count(*) as down, actions.ds as ds from actions, songs where songs.song_id = actions.song_id and action_type = 2  group by songs.artist_id, actions.ds;
# 9145
create view play as select songs.artist_id artist_id, count(*) as plays, actions.ds as ds from actions, songs where songs.song_id = actions.song_id and action_type = 1  group by songs.artist_id, actions.ds;

#7140
create view collect as select songs.artist_id artist_id, count(*) as collect, actions.ds as ds from actions, songs where songs.song_id = actions.song_id and action_type = 3  group by songs.artist_id, actions.ds;






========================================================
#create a user sealyn，and grant it, protect the original dataset
--------------------------------------------------------

create user 'sealyn'@'localhost' identified by 'lyn520';
grant select on xiami.* to 'sealyn'@'localhost';



========================================================
#save the csv data to local
--------------------------------------------------------

select * from playtimes into outfile '/home/sealyn/playtimes.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 

# the data of all artist every day
select actions.ds, count(*) from actions group by actions.ds into outfile '/tmp/allplay.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 

select actions.ds, count(*) from actions where actions.action_type = 0 group by actions.ds into outfile '/tmp/play.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 

select actions.ds, count(*) from actions where actions.action_type = 1 group by actions.ds into outfile '/tmp/download.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 

select actions.ds, count(*) from actions where actions.action_type = 2 group by actions.ds into outfile '/tmp/collect.csv' fields terminated by ',' optionally enclosed by '"' escaped by '"' lines terminated by '\n'; 




========================================================

# query the hot of artist

select song_init_plays from songs 