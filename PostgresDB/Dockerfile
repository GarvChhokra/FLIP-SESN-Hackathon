FROM postgres:latest 
ENV POSTGRES_DB=CanadaVigilanceAdverseReaction 
ENV POSTGRES_USER=postgres 
ENV POSTGRES_PASSWORD=postgres 
COPY init_db.sql /docker-entrypoint-initdb.d/ 
EXPOSE 5432 
