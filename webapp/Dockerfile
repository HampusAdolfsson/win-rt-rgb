FROM node:latest
RUN npm install -g @angular/cli
WORKDIR /app
COPY [ "package.json", "package-lock.json", "./" ]
RUN npm install
COPY "." "."
RUN ng build
EXPOSE 4200
ENTRYPOINT [ "ng", "serve" ]
