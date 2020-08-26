# DataVersionControl_MinIO<br></p>
The objective of this project is used dvc in ML and Data Science projects. For this we'll store large files in MinIO.<br></p>

# MinIO's Setup <br></p>

**1.Install Go**<br></p>
sudo passwd root<br></p>
su root<br></p>

apt update<br></p>
wget -c https://dl.google.com/go/go1.14.2.linux-amd64.tar.gz<br></p>
tar xvf go1.14.2.linux-amd64.tar.gz<br></p>
sudo chown -R root:root ./go<br></p>
sudo mv go /usr/local<br></p>
sudo echo 'export PATH=$PATH:/usr/local/go/bin' >> /etc/profile<br></p>
source /etc/profile<br></p>
rm go1.14.2.linux-amd64.tar.gz<br></p>
go version<br></p>

**2. Install MinIO server** <br></p>
cd ~<br></p>
wget https://dl.min.io/server/minio/release/linux-amd64/minio<br></p>

sudo useradd --system minio --shell /sbin/nologin<br></p>
sudo usermod -L minio<br></p>
sudo chage -E0 minio<br></p>

sudo mv minio /usr/local/bin<br></p>
sudo chmod +x /usr/local/bin/minio<br></p>
sudo chown minio:minio /usr/local/bin/minio<br></p>

sudo touch /etc/default/minio<br></p>
sudo echo 'MINIO_ACCESS_KEY="minio"' >> /etc/default/minio<br></p>
sudo echo 'MINIO_VOLUMES="/usr/local/share/minio/"' >> /etc/default/minio<br></p>
sudo echo 'MINIO_OPTS="-C /etc/minio --address :9000"' >> /etc/default/minio<br></p>
sudo echo 'MINIO_SECRET_KEY="gutteste"' >> /etc/default/minio<br></p>

sudo mkdir /usr/local/share/minio<br></p>
sudo mkdir /etc/minio<br></p>

sudo chown minio:minio /usr/local/share/minio<br></p>
sudo chown minio:minio /etc/minio<br></p>

**3. Install initialize script Minio's systemd**<br></p>
cd ~<br></p>
mkdir minio.service
vim minio.service<br></p>
[Copy and paste the code available in **/setup/minio_service.txt**]<br></p>

sudo mv minio.service /etc/systemd/system<br></p>

**4.Firewall's  configuration**<br></p>
cd ~<br></p>
sudo ufw default deny incoming<br></p>
sudo ufw default allow outgoing<br></p>
sudo ufw allow ssh<br></p>
sudo ufw allow 9000<br></p>
sudo ufw allow http<br></p>
sudo ufw allow https<br></p>
sudo ufw enable<br></p>
sudo ufw allow 80<br></p>
sudo ufw allow 443<br></p>
sudo ufw status verbose<br></p>

**5. Start Minio's service**<br></p>
cd ~<br></p>
sudo systemctl daemon-reload<br></p>
sudo systemctl enable minio<br></p>
sudo systemctl start minio<br></p>
sudo systemctl status minio<br></p>
q (to quit)<br></p>

**6. Install Minio Customer (MC)**<br></p>
cd ~<br></p>
wget https://dl.min.io/client/mc/release/linux-amd64/mc<br></p>
chmod +x mc<br></p>

**7. Set Minio's service with S3 storage (Minio Customer)**<br></p>
cd ~
./mc config host add minio http://127.0.0.1:9000 minio gutteste --api S3v4<br></p>
./mc admin info minio<br></p>
./mc mb minio/storagebr></p>
./mc ls <br></p>
./mc share upload minio/storage/gutteste<br></p>

**8. Set Minio's storage with dvc**<br></p>
export MINIO_ACCESS_KEY="minio"<br></p>
export MINIO_SECRET_KEY="gutteste"<br></p>

Share: curl http://127.0.0.1:9000/storage/ -F x-amz-signature=57ed49f36625ef71333db75b122fa9c9ed583a87899f845df428624c02362ac2 -F bucket=storage -F policy=eyJleHBpcmF0aW9uIjoiMjAyMC0wOC0yOFQxNToyNTo1OS42ODZaIiwiY29uZGl0aW9ucyI6W1siZXEiLCIkYnVja2V0Iiwic3RvcmFnZSJdLFsiZXEiLCIka2V5IiwiZ3V0dGVzdGUiXSxbImVxIiwiJHgtYW16LWRhdGUiLCIyMDIwMDgyMVQxNTI1NTlaIl0sWyJlcSIsIiR4LWFtei1hbGdvcml0aG0iLCJBV1M0LUhNQUMtU0hBMjU2Il0sWyJlcSIsIiR4LWFtei1jcmVkZW50aWFsIiwibWluaW8vMjAyMDA4MjEvdXMtZWFzdC0xL3MzL2F3czRfcmVxdWVzdCJdXX0= -F x-amz-algorithm=AWS4-HMAC-SHA256 -F x-amz-credential=minio/20200821/us-east-1/s3/aws4_request -F x-amz-date=20200821T152559Z -F key=gutteste -F file=@<FILE><br></p>
