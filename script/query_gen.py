import json
import random

with open('data/all_resumes.json', 'r') as f:
    resumes = json.load(f)

skills = ["python", "java", "javascript", "react", "angular", "vue.js", "node.js", "django", "flask", "spring", 
          "html", "css", "sql", "nosql", "mongodb", "postgresql", "mysql", "aws", "azure", "gcp", "docker", 
          "kubernetes", "machine learning", "deep learning", "data analysis", "data visualization", "tensorflow", 
          "pytorch", "scikit-learn", "pandas", "numpy", "r", "tableau", "power bi", "excel", "git", "agile", 
          "scrum", "devops", "ci/cd", "rest api", "graphql", "microservices", "blockchain", "ios", "android", 
          "swift", "kotlin", "c++", "c#", ".net", "ruby", "ruby on rails", "php", "laravel", "symfony"]

education = ["computer science", "software engineering", "data science", "information technology", "electrical engineering", 
             "mechanical engineering", "business administration", "finance", "economics", "mathematics", "statistics", 
             "physics", "biology", "chemistry", "psychology", "marketing", "graphic design", "ux design", "artificial intelligence"]

universities = ["stanford", "mit", "harvard", "berkeley", "carnegie mellon", "caltech", "princeton", "georgia tech", 
                "university of washington", "cornell", "ucla", "university of michigan", "purdue", "university of illinois", 
                "university of texas", "university of wisconsin", "penn state", "ohio state", "virginia tech"]

job_roles = ["software engineer", "data scientist", "web developer", "mobile developer", "devops engineer", "cloud architect", 
             "product manager", "project manager", "ux designer", "ui designer", "data analyst", "business analyst", 
             "machine learning engineer", "ai researcher", "full stack developer", "frontend developer", "backend developer", 
             "qa engineer", "security analyst", "network engineer", "database administrator", "systems administrator"]

queries = []
for _ in range(150):
    query_type = random.choice(["skills", "education", "job_role"])
    if query_type == "skills":
        query = ", ".join(random.sample(skills, k=random.randint(1, 3)))
    elif query_type == "education":
        query = f"{random.choice(education)}, {random.choice(universities)}"
    else:
        query = random.choice(job_roles)
    queries.append(query)

def resume_matches_query(resume, query):
    query_terms = query.lower().split(", ")
    resume_text = json.dumps(resume).lower()
    return all(term in resume_text for term in query_terms)

annotations = {}
for query in queries:
    matching_resumes = [resume["resumeId"][0] for resume in resumes if resume_matches_query(resume, query)]
    if matching_resumes:
        annotations[query] = matching_resumes[:5]

with open('ground_truth_annotations.json', 'w') as f:
    json.dump(annotations, f, indent=2)

print(f"Generated {len(annotations)} annotations.")