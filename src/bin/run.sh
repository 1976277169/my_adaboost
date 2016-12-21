rm -rf log
make -j4 -C ..

if [ -d model ] && [ ! -z "`ls model/|wc -l`" ];then
    mv model model`date +%H%M`
fi


mkdir -p model
./train pos_list.txt neg_list.txt model.dat


