#echo Enter the target typhoon name, channels, start date and end date:
#read name channels start target
name=$1
channels=$2
start=$3
target=$4

mkdir -p $name
Down_Start_Date=$start
Down_End_Date=$target

key=NMSCf0d4fe0d529c4c09b2e920119b659e33

step_day=0
step_hours=0
step_minute=60

contents_down_time=$Down_Start_Date

count=0
while [[ $contents_down_time -le $Down_End_Date ]]; do
save_date_dir=${contents_down_time:0:8}

mkdir -p $name/$save_date_dir

echo $count
wget --no-check-certificate --user-agent Mozilla/4.0 --content-disposition --ignore-length "http://210.125.45.79:9080/api/GK2A/LE1B/$channels/FD/data?date=${contents_down_time}&key=$key" -P ./$name/$save_date_dir/ -a ./LOG/${save_date_dir}_apps.log

contents_down_time=`date -d """${contents_down_time:0:4}-${contents_down_time:4:2}-${contents_down_time:6:2} ${contents_down_time:8:2}:${contents_down_time:10:2}:00 ${step_day} day ${step_hours} hours ${step_minute} minute""" """+%Y%m%d%H%M"""`
count=$(($count+1))

done
			
