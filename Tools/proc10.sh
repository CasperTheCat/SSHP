# 10msData stage 1
python ~/SSHP/Tools/StitchCSV.py xyzv.10.1m.csv xyzv.10.10m.csv xyzv.10.a.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.10.20m.csv xyzv.10.30m.csv xyzv.10.b.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.10.40m.csv xyzv.10.50m.csv xyzv.10.c.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.10.60m.csv xyzv.10.66m.csv xyzv.10.d.csv &

# 15msData stage 1
python ~/SSHP/Tools/StitchCSV.py xyzv.15.1m.csv xyzv.15.10m.csv xyzv.15.a.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.15.20m.csv xyzv.15.30m.csv xyzv.15.b.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.15.40m.csv xyzv.15.50m.csv xyzv.15.c.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.15.60m.csv xyzv.15.66m.csv xyzv.15.d.csv &

#Barrier
wait

# 10ms stage 2
python ~/SSHP/Tools/StitchCSV.py xyzv.10.a.csv xyzv.10.b.csv xyzv.10.ab.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.10.c.csv xyzv.10.d.csv xyzv.10.cd.csv &

# 15ms stage 2
python ~/SSHP/Tools/StitchCSV.py xyzv.15.a.csv xyzv.15.b.csv xyzv.15.ab.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.15.c.csv xyzv.15.d.csv xyzv.15.cd.csv &

# 20ms stage 1
python ~/SSHP/Tools/StitchCSV.py xyzv.20.1m.csv xyzv.20.10m.csv xyzv.20.a.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.20.20m.csv xyzv.20.30m.csv xyzv.20.b.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.20.40m.csv xyzv.20.50m.csv xyzv.20.c.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.20.60m.csv xyzv.20.66m.csv xyzv.20.d.csv &

# Barrier
wait

# 10ms remove stage 1
rm xyzv.10.a.csv
rm xyzv.10.b.csv
rm xyzv.10.c.csv
rm xyzv.10.d.csv

# 15ms remove stage 1
rm xyzv.15.a.csv
rm xyzv.15.b.csv
rm xyzv.15.c.csv
rm xyzv.15.d.csv

# 10ms stage 3
python ~/SSHP/Tools/StitchCSV.py xyzv.10.ab.csv xyzv.10.cd.csv xyzv.10.all.csv &

# 15ms stage 3
python ~/SSHP/Tools/StitchCSV.py xyzv.15.ab.csv xyzv.15.cd.csv xyzv.15.all.csv &

# 20ms stage 2
python ~/SSHP/Tools/StitchCSV.py xyzv.20.a.csv xyzv.20.b.csv xyzv.20.ab.csv &
python ~/SSHP/Tools/StitchCSV.py xyzv.20.c.csv xyzv.20.d.csv xyzv.20.cd.csv &

# Barrier
wait

# 20ms remove stage 1
rm xyzv.20.a.csv
rm xyzv.20.b.csv
rm xyzv.20.c.csv
rm xyzv.20.d.csv

# 10ms remove stage 2
rm xyzv.10.ab.csv
rm xyzv.10.cd.csv

# 15ms remove stage 2
rm xyzv.15.ab.csv
rm xyzv.15.cd.csv

# 20ms stage 3
python ~/SSHP/Tools/StitchCSV.py xyzv.20.ab.csv xyzv.20.cd.csv xyzv.20.all.csv &

# Barrier
wait

# 20ms remove stage 2
rm xyzv.20.ab.csv
rm xyzv.20.cd.csv
