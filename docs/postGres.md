**[Home](index.html) |  [Real-world Examples](examples.html)**

## Setting up of PostGres and PostGIS
[Click here for in-depth manual](https://computingforgeeks.com/how-to-install-postgis-on-ubuntu-linux/)

    sudo apt -y install gnupg2
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
    echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" |sudo tee  /etc/apt/sources.list.d/pgdg.list
### Installation of PostGres
[Click here for installation manual](https://computingforgeeks.com/how-to-install-postgresql-13-on-ubuntu/)

    sudo apt -y install vim bash-completion wget
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
    echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" |sudo tee  /etc/apt/sources.list.d/pgdg.list
    sudo apt update
    sudo apt install postgresql-13 postgresql-client-13
    
    systemctl status postgresql.service
    sudo su - postgres
    psql -c "alter user postgres with password 'StrongAdminP@ssw0rd'"   # change passwd
    psql        # to verify
### Installation of PostGIS

[Click here for more information](https://www.cybertec-postgresql.com/en/postgresql-getting-started-on-ubuntu/)

    sudo apt install postgis postgresql-13-postgis-3
    sudo systemctl restart postgresql
    sudo systemctl status postgresql

### Remote connection settings

    sudo vi /etc/postgresql/13/main/pg_hba.conf    
    # Below "# Accept from anywhere" add the following
        host all all 0.0.0.0/0 md5
        #Database administrative login by Unix domain socket
        local all postgres md5  
    # Save the file and exist

    sudo vi /etc/postgresql/13/main/ postgresql.conf
        listen_addresses = '*'
    # Save and exist

### Removing PostGres and PostGIS

    sudo apt-get --purge remove postgresql   # OR
    sudo apt-get --purge remove postgresql-14  # Version number
