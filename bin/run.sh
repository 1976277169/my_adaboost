
mkdir -p model log/pos log/neg
rm -f model/* log/*.txt log/pos/* log/neg/*

make -C ..

./train ../config model.dat
