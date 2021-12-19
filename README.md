#### Docker Build
```bash
docker build -t comp6231-demo:latest . --build-arg AWS_DEFAULT_REGION=us-east-2 --build-arg AWS_ACCESS_KEY_ID=AKIA2QAXRVAJ5NDDY6H3 --build-arg AWS_SECRET_ACCESS_KEY=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
```

#### Docker Tag
```bash
docker tag comp6231-demo:latest 721603897363.dkr.ecr.us-east-2.amazonaws.com/comp6231-demo:latest
```

#### Docker Run [FOR TESTING]
```bash
docker run -p 9000:8080 comp6231-demo:latest
```

#### Docker Push
```bash
docker push 721603897363.dkr.ecr.us-east-2.amazonaws.com/comp6231-demo:latest
```

#### All-In-One
```bash
docker build -t comp6231-demo:latest . --build-arg AWS_DEFAULT_REGION=us-east-2 --build-arg AWS_ACCESS_KEY_ID=AKIA2QAXRVAJ5NDDY6H3 --build-arg AWS_SECRET_ACCESS_KEY=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 && docker tag comp6231-demo:latest 721603897363.dkr.ecr.us-east-2.amazonaws.com/comp6231-demo:latest && docker push 721603897363.dkr.ecr.us-east-2.amazonaws.com/comp6231-demo:latest
```