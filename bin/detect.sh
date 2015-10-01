if [ -z $1 ] || [ ! -f $1 ] ;then
    echo "Usage: $0 [image]"
    exit;
fi
make -C ..
./detect -m model.dat --scaleStep 1.5 --slideStep 0.08 -i $1 -o out.jpg
