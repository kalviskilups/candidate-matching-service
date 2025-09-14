#!/bin/bash

BASE_URL="http://localhost:8000"

echo
echo "1. API Info"
echo "-----------"
curl -s "$BASE_URL/" | jq '.'

echo
echo "2. Load Sample Data"
echo "-------------------"
curl -X POST "$BASE_URL/ingest/from-json?jobs_file=data/jobs.json&candidates_file=data/candidates.json" | jq '.'

echo
echo "3. Search Dutch Python Jobs in Amsterdam"
echo "-----------------------------------------"
curl -s "$BASE_URL/search/jobs?q=python+developer&k=3&location=Amsterdam&remote=false&skills=python,fastapi&language=nl" | jq '.'

echo
echo "4. Search ML Engineers with Remote Work"
echo "----------------------------------------"
curl -s "$BASE_URL/search/candidates?q=machine+learning&k=3&remote=true&skills=python,tensorflow" | jq '.'

echo
echo "5. Semantic Search - AI and Data Science"
echo "------------------------------------------"
curl -s "$BASE_URL/search/jobs?q=artificial+intelligence+data+science&k=3" | jq '.'

echo
echo "6. DevOps Candidates with Kubernetes"
echo "-------------------------------------"
curl -s "$BASE_URL/search/candidates?q=devops+infrastructure&k=3&skills=kubernetes,docker" | jq '.'

echo
echo "7. Match Candidates to Frontend Job"
echo "------------------------------------"
curl -s "$BASE_URL/match?job_id=job_002&k=3" | jq '.'

echo
echo "8. Match Candidates to ML Job"
echo "------------------------------"
curl -s "$BASE_URL/match?job_id=job_123&k=3" | jq '.'

echo
echo "9. Security Jobs in Dutch"
echo "--------------------------"
curl -s "$BASE_URL/search/jobs?q=security+beveiliging&k=2&language=nl" | jq '.'

echo
echo "10. Senior Engineers in Netherlands"
echo "------------------------------------"
curl -s "$BASE_URL/search/candidates?q=senior+engineer&k=3&location=Netherlands&remote=false" | jq '.'

echo
echo "All example calls completed!"
echo "Visit http://localhost:8000/docs for interactive API documentation."
