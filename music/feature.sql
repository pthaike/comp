# 每首歌曲每天的听歌量，有多少首歌曲听歌量大于一个阈值

select count(*) from actions, songs where actions.song_id = songs.song_id and songs.artist_id = '6bb4c3bbdb6f5a96d643320c6b6005f5' group by  songs.song_id having count(*) > avg(count(*));

# 前一天的收藏量，下载量，听歌量

# 艺人歌曲top-k 的听歌量 


select * from actions where actions.user_id = '319786cabc91dee14fd19dc2a566bf52';

select * from ac where ds between '2015-03-15' and '2015-03-16';

select songs.song_id, count(*) from ac, songs where ac.song_id = songs.song_id and ds = '2015-03-16' and songs.artist_id = 'd773376e46311393cd89994bf9a93043' group by songs.song_id order by count(*) desc limit 100;



