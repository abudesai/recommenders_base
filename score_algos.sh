#! /usr/bin/bash

unset image
unset algo
unset volume

while getopts i:a:v: flag
do
    case "${flag}" in
        i) image=${OPTARG};;
        a) algo=${OPTARG};;
        v) volume=${OPTARG};;
        *)
            echo 'Error in command line parsing' >&2
            exit 1
    esac
done

shift "$(( OPTIND - 1 ))"

if [ -z "$image" ] ; then
        echo 'Error: Missing -i flag for image to pull' >&2
        exit 1
fi

if [ -z "$algo" ] ; then
        echo 'Error: Missing -a flag for algo to pull' >&2
        exit 1
fi

if [ -z "$volume" ]; then
        echo 'Error: Missing -v flag for volume name' >&2
        exit 1
fi


# echo "image: $image";
# echo "algo: $algo";
# echo "volume: $volume";


#---------------------------------------------------------------------------

host_vol_path="$(pwd)/${volume}"
container_vol_path="/app/${volume}"

echo "host_vol_path: $host_vol_path"
echo "container_vol_path: $container_vol_path"

container_name="$algo"

# stop and remove container if it exists
docker stop $container_name || true && docker rm $container_name || true

# now run the container
docker run -d -p 3000:3000 -v $host_vol_path:$container_vol_path --name $container_name $image

# run training
docker exec $container_name python train.py

#run test predictions 
docker exec $container_name python predict.py

#run scoring 
docker exec $container_name python score.py


#---------------------------------------------------------------------------
# stop and remove the container
docker stop $container_name
docker rm $container_name

# # remove the image
# docker rmi $image


#---------------------------------------------------------------------------
