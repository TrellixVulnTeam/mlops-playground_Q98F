# export AWS_PROFILE= ...

# Specify a repository name.
repository_name=sagemaker-category-encoders

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration. (default to us-east-1 if none defined)
region=$(aws configure get region)

full_name="${account}.dkr.ecr.${region}.amazonaws.com/${repository_name}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${repository_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${repository_name}" > /dev/null
fi

$(aws ecr get-login --region ${region} --no-include-email)
$(aws ecr get-login --registry-ids 683313688378 --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${repository_name} .
docker tag ${repository_name} ${full_name}
docker push ${full_name}
