FROM node:22

WORKDIR /frontend

COPY ./package*.json ./

RUN npm install

COPY . .

EXPOSE 3000
ENTRYPOINT [ "npm" ]
CMD ["start"]