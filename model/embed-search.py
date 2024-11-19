from gensim.models import Word2Vec
import faiss
import numpy as np
from functools import lru_cache
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import math
import spacy
import json
import re
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

class EmbeddingProcessor:
    def __init__(self, word2vec_model, spacy_nlp, vectorizer, cache_size=1000):
        self.word2vec_model = word2vec_model
        self.nlp = spacy_nlp
        self.vectorizer = vectorizer
        self.logger = logging.getLogger(__name__)
        
        # Embedding dimension configuration
        self.word2vec_dim = 300
        self.spacy_dim = self.nlp.vocab.vectors_length
        
        # Weights for different embedding types
        self.weights = {
            'word2vec': 0.4,
            'spacy': 0.3,
            'tfidf': 0.3
        }
        
        # Cache size configuration
        self.cache_size = cache_size

    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """
        Cached text preprocessing with more robust cleaning
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text

    @lru_cache(maxsize=1000)
    def get_word2vec_embedding(self, text: str) -> np.ndarray:
        """
        Get Word2Vec embedding with improved error handling
        """
        try:
            words = text.split()
            valid_words = [w for w in words if w in self.word2vec_model.wv]
            
            if not valid_words:
                return np.zeros(self.word2vec_dim)
            
            embeddings = [self.word2vec_model.wv[w] for w in valid_words]
            return np.mean(embeddings, axis=0)
        
        except Exception as e:
            self.logger.error(f"Word2Vec embedding error: {e}")
            return np.zeros(self.word2vec_dim)

    @lru_cache(maxsize=1000)
    def get_spacy_embedding(self, text: str) -> np.ndarray:
        """
        Get SpaCy embedding with improved error handling
        """
        try:
            doc = self.nlp(text)
            if doc.has_vector:
                vector = doc.vector
                if len(vector) > self.word2vec_dim:
                    return vector[:self.word2vec_dim]
                elif len(vector) < self.word2vec_dim:
                    return np.pad(vector, (0, self.word2vec_dim - len(vector)))
                return vector
            return np.zeros(self.word2vec_dim)
        
        except Exception as e:
            self.logger.error(f"SpaCy embedding error: {e}")
            return np.zeros(self.word2vec_dim)

    @lru_cache(maxsize=1000)
    def get_tfidf_embedding(self, text: str) -> np.ndarray:
        """
        Get TF-IDF embedding with simplified approach
        """
        try:
            tfidf_vector = self.vectorizer.transform([text]).toarray().flatten()
            
            # Simple dimension reduction: take average of chunks
            if len(tfidf_vector) > self.word2vec_dim:
                n_chunks = len(tfidf_vector) // self.word2vec_dim
                if n_chunks > 0:
                    chunks = np.array_split(tfidf_vector, self.word2vec_dim)
                    return np.array([chunk.mean() for chunk in chunks])
                else:
                    # If vector is too short, pad with zeros
                    return np.pad(tfidf_vector, (0, self.word2vec_dim - len(tfidf_vector)))
            elif len(tfidf_vector) < self.word2vec_dim:
                # Pad with zeros if vector is too short
                return np.pad(tfidf_vector, (0, self.word2vec_dim - len(tfidf_vector)))
            return tfidf_vector
            
        except Exception as e:
            self.logger.error(f"TF-IDF embedding error: {e}")
            return np.zeros(self.word2vec_dim)

    def combine_embeddings(self, 
                         word2vec_emb: np.ndarray, 
                         spacy_emb: np.ndarray, 
                         tfidf_emb: np.ndarray) -> np.ndarray:
        """
        Combine embeddings with simplified approach
        """
        # Ensure all embeddings have correct dimensions
        word2vec_emb = self._ensure_dimension(word2vec_emb)
        spacy_emb = self._ensure_dimension(spacy_emb)
        tfidf_emb = self._ensure_dimension(tfidf_emb)
        
        # Weighted combination
        combined_emb = (
            self.weights['word2vec'] * word2vec_emb +
            self.weights['spacy'] * spacy_emb +
            self.weights['tfidf'] * tfidf_emb
        )
        
        return combined_emb

    def _ensure_dimension(self, vector: np.ndarray) -> np.ndarray:
        """
        Ensure vector has correct dimension
        """
        if len(vector) > self.word2vec_dim:
            return vector[:self.word2vec_dim]
        elif len(vector) < self.word2vec_dim:
            return np.pad(vector, (0, self.word2vec_dim - len(vector)))
        return vector

    def safe_normalize(self, vector: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Safe vector normalization
        """
        norm = np.linalg.norm(vector)
        if norm < epsilon:
            return vector
        return vector / norm

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Get combined embedding with improved error handling"""
        if not text or not isinstance(text, str):
            return np.zeros(300)  # Use consistent dimension
            
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return np.zeros(300)
            
        try:
            # Get individual embeddings
            word2vec_emb = self.get_word2vec_embedding(processed_text)
            spacy_emb = self.get_spacy_embedding(processed_text)
            tfidf_emb = self.get_tfidf_embedding(processed_text)
            
            # Combine embeddings
            combined_emb = self.combine_embeddings(word2vec_emb, spacy_emb, tfidf_emb)
            
            # Normalize if requested
            if normalize:
                combined_emb = self.safe_normalize(combined_emb)
                
            return combined_emb
        except Exception as e:
            logging.error(f"Error in get_embedding: {e}")
            return np.zeros(300)
    
class ResumeSearchEngine:
    def __init__(self):
        self.resumes = self.load_resumes()
        self.nlp = spacy.load('en_core_web_md')
        self.vectorizer = TfidfVectorizer(stop_words='english')
        all_resume_texts = [self.preprocess_resume(resume) for resume in self.resumes]
        self.vectorizer.fit(all_resume_texts)
        self.word2vec_model = self.train_word2vec()
        self.embedding_processor = EmbeddingProcessor(
                    word2vec_model=self.word2vec_model,
                    spacy_nlp=self.nlp,
                    vectorizer=self.vectorizer
                )       
        self.faiss_index = self.build_faiss_index()

    def load_resumes(self) -> List[Dict]:
        """Load resumes from JSON file"""
        file_path = os.path.join('data', 'all_resumes.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: Could not find file at {file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            return []

    def train_word2vec(self) -> Word2Vec:
        """Train Word2Vec model on resume corpus"""
        sentences = []
        for resume in self.resumes:
            text_data = []
            text_data.extend([skill['skillName'].lower() for skill in resume.get('skills', [])])
            for edu in resume.get('education', []):
                text_data.extend([edu.get('degree', ''), edu.get('major', '')])
            for exp in resume.get('workExperience', []):
                text_data.extend(self.preprocess_text(exp.get('description', '')).split())
            sentences.append(text_data)
        return Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)

    def create_tfidf_matrix(self):
        """Create TF-IDF matrix for all resumes"""
        documents = [self.preprocess_resume(resume) for resume in self.resumes]
        return self.vectorizer.fit_transform(documents)

    def build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        embeddings = self.get_all_embeddings()
        
        # Ensure embeddings are not empty
        if len(embeddings) == 0:
            raise ValueError("No embeddings generated for indexing")
            
        # Convert to correct format
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Create index with correct dimension
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Changed to L2 distance
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)  # Proper normalization
        
        # Add to index
        index.add(embeddings)
        
        return index

    def get_embedding(self, text):
        return self.embedding_processor.get_embedding(text)

    def get_all_embeddings(self):
        """Get embeddings for all resumes"""
        # Process resumes in smaller batches to manage memory
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(self.resumes), batch_size):
            batch = self.resumes[i:i + batch_size]
            batch_embeddings = []
            
            for resume in batch:
                embedding = self.get_resume_embedding(resume)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                else:
                    batch_embeddings.append(np.zeros(300, dtype=np.float32))
            
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array and ensure float32 type
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        return embeddings_array

    def get_resume_embedding(self, resume: Dict) -> np.ndarray:
        """Get embedding for a single resume"""
        text = self.preprocess_resume(resume)
        if not text:
            return np.zeros(300, dtype=np.float32)
        
        embedding = self.embedding_processor.get_embedding(text)
        
        # Ensure correct type and shape
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Ensure 1D array
        if len(embedding.shape) > 1:
            embedding = embedding.flatten()
        
        return embedding.astype(np.float32)

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-z0-9\s\-\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def preprocess_resume(self, resume: Dict) -> str:
        """Improved resume text processing"""
        text_parts = []
        
        # Add skills
        skills = [skill['skillName'] for skill in resume.get('skills', [])]
        text_parts.extend(skills)
        
        # Add work experience
        for exp in resume.get('workExperience', []):
            text_parts.append(exp.get('role', ''))
            text_parts.append(exp.get('description', ''))
            
        # Add education
        for edu in resume.get('education', []):
            text_parts.append(edu.get('degree', ''))
            text_parts.append(edu.get('major', ''))
            
        # Join all parts and preprocess
        text = ' '.join(filter(None, text_parts))
        return self.preprocess_text(text)

    @lru_cache(maxsize=100)
    def parallel_comprehensive_search(self, queries: Tuple[str, ...], threshold: float = 0.6, num_processes: int = 4) -> List[Dict]:
        """
        Parallelized comprehensive search across multiple resume fields
        
        Args:
            queries (Tuple[str, ...]): Tuple of search terms
            threshold (float): Minimum similarity threshold
            num_processes (int): Number of parallel processes to use
            
        Returns:
            List of matching resumes with match details
        """
        # Convert tuple to list for processing
        query_list = list(queries)
        
        # Split resumes into chunks for parallel processing
        chunks = np.array_split(self.resumes, num_processes)
        
        # Create a pool of worker processes
        with Pool(processes=num_processes) as pool:
            # Map the search function to each chunk of resumes
            results = pool.starmap(
                self._chunk_comprehensive_search,
                [(chunk.tolist(), query_list, threshold) for chunk in chunks]
            )
        
        # Combine and sort results from all processes
        combined_results = [item for sublist in results for item in sublist]
        return sorted(combined_results, key=lambda x: x['match_score'], reverse=True)

    def _chunk_comprehensive_search(self, resumes: List[Dict], queries: List[str], threshold: float) -> List[Dict]:
        """
        Comprehensive search for a chunk of resumes (to be used by each process)
        """
        results = []
        for resume in resumes:
            matches = []
            sections_to_search = [
                {'field': 'skills', 'items': [skill['skillName'].lower() for skill in resume.get('skills', [])]},
                {'field': 'work_description', 'items': [self.preprocess_text(exp.get('description', '').lower()) for exp in resume.get('workExperience', [])]},
                {'field': 'work_roles', 'items': [self.preprocess_text(exp.get('role', '').lower()) for exp in resume.get('workExperience', [])]},
                {'field': 'education', 'items': [self.preprocess_text(f"{edu.get('degree', '')} {edu.get('major', '')}").lower() for edu in resume.get('education', [])]}
            ]

            resume_score = 0
            total_matches = 0

            for query in queries:
                query_embedding = self.get_embedding(query)
                query_match = False

                for section in sections_to_search:
                    for item in section['items']:
                        fuzzy_score = fuzz.ratio(query.lower(), item) / 100
                        item_embedding = self.get_embedding(item)
                        semantic_similarity = cosine_similarity([query_embedding], [item_embedding])[0][0]
                        combined_score = (fuzzy_score + semantic_similarity) / 2

                        if combined_score >= threshold:
                            matches.append({
                                'query': query,
                                'match': item,
                                'section': section['field'],
                                'score': combined_score
                            })
                            query_match = True
                            resume_score += combined_score
                            total_matches += 1

                if not query_match:
                    resume_score -= 0.2

            if matches:
                results.append({
                    'resume': resume,
                    'matches': matches,
                    'match_score': resume_score / max(total_matches, 1)
                })

        return results

    def search_skills(self, terms: List[str], threshold: float = 0.75) -> List[Dict]:
        """Search for resumes based on skills"""
        results = []
        for resume in self.resumes:
            resume_skills = set(skill['skillName'].lower() for skill in resume.get('skills', []))
            matches = []
            for term in terms:
                for skill in resume_skills:
                    similarity = fuzz.ratio(term.lower(), skill) / 100
                    if similarity >= threshold:
                        matches.append({
                            'query': term,
                            'match': skill,
                            'section': 'skills',
                            'score': similarity
                        })
            if matches:
                results.append({
                    'resume': resume,
                    'matches': matches,
                    'match_score': sum(match['score'] for match in matches) / len(matches)
                })
        return sorted(results, key=lambda x: x['match_score'], reverse=True)

    def search_education(self, terms: List[str], threshold: float = 0.8) -> List[Dict]:
        """Search for resumes based on education"""
        results = []
        for resume in self.resumes:
            education = resume.get('education', [])
            matches = []
            for term in terms:
                for edu in education:
                    edu_text = f"{edu.get('degree', '')} {edu.get('major', '')} {edu.get('school', '')}"
                    similarity = fuzz.partial_ratio(term.lower(), edu_text.lower()) / 100
                    if similarity >= threshold:
                        matches.append({
                            'query': term,
                            'match': edu_text,
                            'section': 'education',
                            'score': similarity
                        })
            if matches:
                results.append({
                    'resume': resume,
                    'matches': matches,
                    'match_score': sum(match['score'] for match in matches) / len(matches)
                })
        return sorted(results, key=lambda x: x['match_score'], reverse=True)

    def search_experience(self, terms: List[str], min_years: Optional[int] = None) -> List[Dict]:
        """Search for resumes based on work experience"""
        results = []
        for resume in self.resumes:
            work_experience = resume.get('workExperience', [])
            matches = []
            total_years = 0
            for exp in work_experience:
                exp_text = f"{exp.get('role', '')} {exp.get('company', '')} {exp.get('description', '')}"
                start_year = int(exp.get('start_date', '0')[:4])
                end_year = int(exp.get('end_date', '0')[:4]) if exp.get('end_date') != 'Present' else 2024
                years = end_year - start_year
                total_years += years
                for term in terms:
                    similarity = fuzz.partial_ratio(term.lower(), exp_text.lower()) / 100
                    if similarity >= 0.6:  # Using a lower threshold for experience
                        matches.append({
                            'query': term,
                            'match': exp_text,
                            'section': 'experience',
                            'score': similarity,
                            'years': years
                        })
            if matches and (min_years is None or total_years >= min_years):
                results.append({
                    'resume': resume,
                    'matches': matches,
                    'match_score': sum(match['score'] for match in matches) / len(matches),
                    'total_years': total_years
                })
        return sorted(results, key=lambda x: (x['total_years'], x['match_score']), reverse=True)
    
    def search(self, query: Any, k: int = 10) -> List[Dict]:
        """Unified search method that handles different query types"""
        if isinstance(query, str):
            # Convert single query string to tuple before passing to parallel_comprehensive_search
            return self.parallel_comprehensive_search((query,), threshold=0.6)
        elif isinstance(query, dict) and 'type' in query:
            query_type = query['type']
            if query_type == "skills":
                return self.search_skills(query['terms'], threshold=query.get('threshold', 0.75))
            elif query_type == "education":
                return self.search_education(query['terms'], threshold=query.get('threshold', 0.8))
            elif query_type == "experience":
                return self.search_experience(query['terms'], min_years=query.get('min_years', None))
            else:
                raise ValueError(f"Invalid query type: {query_type}")
        else:
            # Convert query to tuple for hashability
            return self.parallel_comprehensive_search((str(query),), threshold=0.6)

class AdvancedResumeRanker:
    def __init__(self, search_engine):
        self.search_engine = search_engine
    
    def calculate_advanced_relevance_score(self, resume: Dict, queries: List[str]) -> Dict:
        """
        Calculate a multi-dimensional relevance score with granular criteria
        
        Scoring Dimensions:
        1. Query Match Depth
        2. Professional Experience Alignment
        3. Skill Precision
        4. Career Trajectory Consistency
        5. Contextual Relevance
        """
        scores = {
            'query_match_depth': 0,
            'experience_alignment': 0,
            'skill_precision': 0,
            'career_consistency': 0,
            'contextual_relevance': 0
        }
        
        # 1. Query Match Depth
        total_matches = len(resume.get('matches', []))
        unique_match_sections = len(set(match['section'] for match in resume.get('matches', [])))
        scores['query_match_depth'] = (
            total_matches * 0.5 + 
            unique_match_sections * 0.3
        )
        
        # 2. Professional Experience Alignment
        work_experiences = resume.get('workExperience', [])
        if work_experiences:
            # Check depth and recency of relevant experiences
            relevant_experiences = [
                exp for exp in work_experiences 
                if any(query.lower() in exp.get('description', '').lower() 
                       or query.lower() in exp.get('role', '').lower() 
                       for query in queries)
            ]
            
            # Weigh recent and longer experiences more
            experience_score = sum(
                (1 - abs(2024 - int(exp.get('end_date', 2024))) / 10) *  # Recency
                (min(5, (int(exp.get('end_date', 2024)) - int(exp.get('start_date', 0)))) / 5)  # Duration
                for exp in relevant_experiences
            )
            
            scores['experience_alignment'] = experience_score / max(len(relevant_experiences), 1)
        
        # 3. Skill Precision
        resume_skills = set(skill['skillName'].lower() for skill in resume.get('skills', []))
        query_skills = set(query.lower() for query in queries)
        skill_overlap = len(resume_skills.intersection(query_skills))
        scores['skill_precision'] = skill_overlap / max(len(query_skills), 1)
        
        # 4. Career Trajectory Consistency
        # Look for progressive roles or consistent domain expertise
        roles = [exp.get('role', '').lower() for exp in work_experiences]
        unique_roles = len(set(roles))
        role_consistency_score = 1 / (1 + math.log(unique_roles + 1))
        scores['career_consistency'] = role_consistency_score
        
        # 5. Contextual Relevance
        # Consider education and additional context
        education_matches = sum(
            1 for edu in resume.get('education', []) 
            if any(query.lower() in str(edu).lower() for query in queries)
        )
        scores['contextual_relevance'] = education_matches * 0.2
        
        # Weighted Aggregate Score
        weights = {
            'query_match_depth': 0.3,
            'experience_alignment': 0.25,
            'skill_precision': 0.2,
            'career_consistency': 0.15,
            'contextual_relevance': 0.1
        }
        
        # Calculate final relevance score
        final_score = sum(
            scores[dim] * weights[dim] 
            for dim in scores
        )
        
        return {
            'scores': scores,
            'final_relevance_score': final_score
        }
    
    def filter_and_rank_candidates(
        self, 
        search_results: List[Dict], 
        queries: List[str], 
        top_n: int = 10,
        relevance_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Advanced candidate filtering and ranking
        
        Args:
            search_results: Raw search results
            queries: Original search queries
            top_n: Number of top candidates to return
            relevance_threshold: Minimum relevance score to consider
        
        Returns:
            Ranked and filtered candidate list
        """
        # Enhance each resume with detailed relevance analysis
        ranked_candidates = []
        for result in search_results:
            relevance_analysis = self.calculate_advanced_relevance_score(result, queries)
            
            # Only consider candidates above relevance threshold
            if relevance_analysis['final_relevance_score'] >= relevance_threshold:
                ranked_candidates.append({
                    'resume': result['resume'],
                    'matches': result['matches'],
                    'relevance_analysis': relevance_analysis
                })
        
        # Sort by final relevance score
        ranked_candidates.sort(
            key=lambda x: x['relevance_analysis']['final_relevance_score'], 
            reverse=True
        )
        
        return ranked_candidates[:top_n]
        
def main():
    search_engine = ResumeSearchEngine()
    ranker = AdvancedResumeRanker(search_engine)

    while True:
        print("\nAdvanced Resume Candidate Search")
        print("1. General Search")
        print("2. Skills Search")
        print("3. Education Search")
        print("4. Experience Search")
        print("5. Quit")

        choice = input("Enter your choice (1-5): ")

        if choice == '5':
            break

        if choice == '1':
            query = input("Enter your general search query: ")
            results = search_engine.search(query)
            terms = [query]  # For ranking purposes
        elif choice in ['2', '3', '4']:
            terms = input("Enter search terms (comma-separated): ").split(',')
            terms = [term.strip() for term in terms]
            
            if choice == '2':
                threshold = float(input("Enter threshold (0.0-1.0, default 0.75): ") or 0.75)
                query = {"type": "skills", "terms": terms, "threshold": threshold}
            elif choice == '3':
                threshold = float(input("Enter threshold (0.0-1.0, default 0.8): ") or 0.8)
                query = {"type": "education", "terms": terms, "threshold": threshold}
            elif choice == '4':
                min_years = int(input("Enter minimum years of experience (default 0): ") or 0)
                query = {"type": "experience", "terms": terms, "min_years": min_years}
            
            results = search_engine.search(query)
        else:
            print("Invalid choice. Please try again.")
            continue

        ranked_results = ranker.filter_and_rank_candidates(results, terms)

        print(f"\nTop {len(ranked_results)} Candidates:")
        for i, result in enumerate(ranked_results, 1):
            try:
                resume_name = result['resume']['personalInformation'].get('name', 'Unknown')
            except KeyError:
                resume_name = 'Unknown'
            relevance_score = result['relevance_analysis']['final_relevance_score']
            print(f"{i}. {resume_name} (Relevance: {relevance_score:.2f})")
            print(" Matches:")
            for match in result['matches'][:3]:  # Show top 3 matches
                print(f" - {match['query']} in {match['section']} (Score: {match['score']:.2f})")
            print(" Relevance Analysis:")
            for dimension, score in result['relevance_analysis']['scores'].items():
                print(f" - {dimension.replace('_', ' ').title()}: {score:.2f}")
            print()

if __name__ == "__main__":
    main()