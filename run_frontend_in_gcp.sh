INSTANCE_NAME="instance-1"
REGION=us-central1
ZONE=us-central1-c
PROJECT_NAME="ir-2022-my"
IP_NAME="$PROJECT_NAME-ip"
GOOGLE_ACCOUNT_NAME="eitag" # without the @post.bgu.ac.il or @gmail.com part

# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
gcloud config set project ir-2022-my
gcloud config set compute/zone us-central1-c

# 1. Set up public IP
gcloud compute addresses create ir-2022-my-ip --project=ir-2022-my --region=us-central1
gcloud compute addresses list
# note the IP address printed above, that's your extrenal IP address.
# Enter it here: 
INSTANCE_IP="34.136.190.17"

# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 
  --allow tcp:8080 
  --source-ranges 0.0.0.0/0 
  --target-tags http-server

# 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
gcloud compute instances create instance-1 
  --zone=us-central1-c
  --machine-type=n1-standard-16
  --network-interface=address=34.136.190.17,network-tier=PREMIUM,subnet=default
  --metadata-from-file startup-script=startup_script_gcp.sh
  --scopes=https://www.googleapis.com/auth/cloud-platform
  --tags=http-server
# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
gcloud compute instances tail-serial-port-output instance-1 --zone us-central1-c

# 5. SSH to your VM and start the app
gcloud compute ssh eitag@instance-1
#create app dir that holds all the files
mkdir app
cd app
# 4. Secure copy your app to the VM
gcloud compute scp search_frontend.py eitag@instance-1:/home/eitag
gcloud compute scp *.py eitag@instance-1:/home/eitag/app
gcloud compute scp *.json eitag@instance-1:/home/eitag/app
#this dir is only for testing
mkdir indices_stat 
cd indices_stat


gsutil -m cp -r gs://testing-bucketing/postings_gcp/regular-index/text-index



python3 search_frontend.py

################################################################################
# Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q instance-1
# make sure there are no lingering instances
gcloud compute instances list
# delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080
# delete external addresses

gcloud compute addresses delete ir-2022-my-ip --region us-central1

#gcloud compute addresses delete -q 34.135.192.199 --region us-central1

gcloud compute disks create disk-1 --project=ir-2022-my --type=pd-standard --size=150GB --zone=us-central1-c