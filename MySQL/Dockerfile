FROM mysql:8

ENV MYSQL_ROOT_PASSWORD=rootpassword

COPY ./init_db /docker-entrypoint-initdb.d
EXPOSE 3306
CMD ["mysqld"]
