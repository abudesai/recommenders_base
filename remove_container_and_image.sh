#! /usr/bin/bash

unset image
unset algo

while getopts i:a: flag
do
    case "${flag}" in
        i) image=${OPTARG};;
        a) algo=${OPTARG};;
        *)
            echo 'Error in command line parsing' >&2
            exit 1
    esac
done

shift "$(( OPTIND - 1 ))"

if [ -z "$image" ] ; then
        echo 'Error: Missing -i flag for image to remove' >&2
        exit 1
fi

if [ -z "$algo" ] ; then
        echo 'Error: Missing -a flag for algo_name' >&2
        exit 1
fi

# echo "image: $image";
# echo "algo: $algo";

#---------------------------------------------------------------------------

container_name="$algo"

# stop and remove container if it exists
docker stop $container_name || true && docker rm $container_name || true

# remove the image
docker rmi $image || true


#---------------------------------------------------------------------------
