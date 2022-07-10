#! /bin/bash

cmd=$1

if [ $(systemctl is-active docker) == "inactive" ] ; then
  printf "Docker's service is not active, use cli:\n systemctl start docker \n"

else
  if [ -r ./docker/.env ] ; then
    if [ $cmd == "start" ] ; then
      docker-compose --env-file ./docker/.env up

    elif [ $cmd == "stop" ] ; then
      docker-compose --env-file ./docker/.env down

    else
      printf "server.sh: '%s' is not a valid command \nuse: start, stop \n" "$cmd"
    fi

  else
    echo "Can't can't run docker-compose file: no docker variables, './docker/.env' is not readable or does not exist"
  fi
fi
