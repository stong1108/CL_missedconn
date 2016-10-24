
## Setting up on AWS EC2 Instance (Ubuntu 14.04)

### Install Anaconda
```
wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh
```

Run `bash` on Anaconda package.

### Install Postgres
Link to instructions:
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-14-04
```
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo -i -u postgres
createuser --superuser ubuntu
createdb ubuntu
createdb cl_missedconn
```

Change engine address in `manage_db.py`:
Use vim command `:%s/old/new/g` to replace `stong@localhost:5432` (old) with nothing (new), so what used to be
```
postgresql://stong@localhost:5432/cl_missedconn
```
now becomes
```
postgresql:///cl_missedconn
```

Use `quit` to get back to `ubuntu` user and check for access with `psql`.

### Grab things from Github
```
sudo apt-get install git
git clone https://github.com/stong1108/cl_missedconn
git clone https://github.com/stong1108/vimconfig
cd vimconfig
bash install.sh
```

### Install packages
```
pip install langdetect
pip install HTMLParser
pip install unidecode
sudo aptitude install python-dev libpq-dev
pip install psycopg2
```
### Populate db
`psql cl_missedconn < *.sql`
