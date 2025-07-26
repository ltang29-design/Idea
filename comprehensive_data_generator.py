import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class ComprehensiveProfileGenerator:
    def __init__(self):
        # Real names with diversity
        self.real_names = {
            'western': [
                "John Smith", "Emily Chen", "Michael Brown", "Sophie Lee", "David Kim", 
                "Anna Ivanova", "James Wilson", "Sarah Johnson", "Robert Davis", "Lisa Anderson",
                "William Taylor", "Jennifer Martinez", "Christopher Garcia", "Amanda Rodriguez",
                "Daniel Lopez", "Jessica White", "Matthew Thompson", "Nicole Clark", "Joshua Lewis",
                "Stephanie Hall", "Andrew Young", "Rachel Allen", "Kevin King", "Megan Scott",
                "Brian Green", "Lauren Baker", "Steven Adams", "Ashley Nelson", "Timothy Carter",
                "Brittany Mitchell", "Jeffrey Perez", "Samantha Roberts", "Ryan Turner", "Heather Phillips",
                "Gary Campbell", "Amber Evans", "Eric Edwards", "Melissa Collins", "Jacob Stewart",
                "Danielle Morris", "Nathan Rogers", "Tiffany Reed", "Adam Cook", "Crystal Bailey",
                "Mark Cooper", "Erica Richardson", "Donald Cox", "Monica Ward", "Kenneth Torres",
                "Jacqueline Peterson", "Ronald Gray", "Christina Ramirez", "Anthony James", "Catherine Watson"
            ],
            'asian': [
                "Wei Zhang", "Yuki Tanaka", "Jin Park", "Mei Lin", "Hiroshi Yamamoto",
                "Soo-Jin Kim", "Xiaoli Wang", "Kenji Sato", "Min-ji Lee", "Ling Chen",
                "Takashi Ito", "Ji-eun Park", "Feng Liu", "Yumi Nakamura", "Seung-ho Choi",
                "Xiaoyu Li", "Kazuki Suzuki", "Hye-jin Kim", "Ming Zhao", "Akira Tanaka",
                "Jung-soo Lee", "Yan Zhang", "Shinji Watanabe", "Eun-ji Park", "Tao Wang",
                "Yusuke Kobayashi", "Min-seok Kim", "Li Wei", "Haruka Sato", "Jae-hyun Lee"
            ],
            'indian': [
                "Priya Patel", "Raj Sharma", "Anjali Singh", "Vikram Malhotra", "Kavita Reddy",
                "Arjun Gupta", "Meera Iyer", "Sanjay Verma", "Divya Kapoor", "Rahul Joshi",
                "Neha Desai", "Amit Kumar", "Pooja Sharma", "Vivek Singh", "Riya Patel",
                "Karan Malhotra", "Anita Reddy", "Deepak Gupta", "Sunita Iyer", "Mohan Verma"
            ],
            'hispanic': [
                "Maria Rodriguez", "Carlos Martinez", "Ana Garcia", "Jose Lopez", "Isabella Perez",
                "Miguel Torres", "Sofia Ramirez", "Diego Gonzalez", "Valentina Morales", "Alejandro Silva",
                "Camila Herrera", "Luis Vargas", "Gabriela Castro", "Fernando Ruiz", "Lucia Mendoza",
                "Ricardo Ortega", "Elena Jimenez", "Hector Moreno", "Adriana Vega", "Roberto Cruz"
            ]
        }
        
        # Fake names (obviously fake)
        self.fake_names = [
            "James Bond", "Jane Doe", "Chris P Bacon", "Sue Permann", "Joe King", 
            "Mary Christmas", "Ben Dover", "Anita Bath", "Ima Hogg", "Justin Case",
            "Eileen Dover", "Hugh Jass", "Dewey Cheatham", "Ivana Tinkle", "Moe Lester",
            "Anita Mann", "Dixon Butts", "Hugh G. Rection", "Ivana Humpalot", "Mike Hunt",
            "Anita Dick", "Dixon Cider", "Hugh Janus", "Ivana Trump", "Mike Oxlong",
            "Anita Cock", "Dixon Butts", "Hugh Jass", "Ivana Tinkle", "Mike Hunt"
        ]
        
        # Real companies by industry
        self.real_companies = {
            'tech': [
                "Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", "Tesla", "Uber",
                "Airbnb", "Salesforce", "Oracle", "IBM", "Intel", "Adobe", "Cisco", "NVIDIA",
                "Spotify", "Twitter", "LinkedIn", "Zoom", "Slack", "Dropbox", "Palantir",
                "Stripe", "Square", "Shopify", "Twilio", "Datadog", "MongoDB", "Snowflake"
            ],
            'finance': [
                "JPMorgan Chase", "Bank of America", "Wells Fargo", "Goldman Sachs", "Morgan Stanley",
                "Citigroup", "American Express", "BlackRock", "Visa", "Mastercard", "PayPal",
                "Charles Schwab", "Fidelity", "State Street", "PNC Financial", "US Bancorp",
                "Capital One", "TD Bank", "HSBC", "Deutsche Bank", "Credit Suisse", "UBS"
            ],
            'healthcare': [
                "Johnson & Johnson", "Pfizer", "UnitedHealth Group", "Anthem", "Aetna",
                "Cigna", "Humana", "CVS Health", "Walgreens", "McKesson", "AmerisourceBergen",
                "Cardinal Health", "Bristol-Myers Squibb", "Merck", "Amgen", "Gilead Sciences",
                "Biogen", "Regeneron", "Moderna", "Novartis", "Roche", "Sanofi"
            ],
            'consulting': [
                "McKinsey & Company", "Bain & Company", "Boston Consulting Group", "Deloitte",
                "PwC", "EY", "KPMG", "Accenture", "Capgemini", "IBM Consulting", "Booz Allen Hamilton",
                "Oliver Wyman", "Strategy&", "Roland Berger", "A.T. Kearney", "L.E.K. Consulting"
            ],
            'retail': [
                "Walmart", "Target", "Costco", "Home Depot", "Lowe's", "Best Buy", "Macy's",
                "Kohl's", "Nordstrom", "TJ Maxx", "Ross Stores", "Dollar General", "Dollar Tree",
                "CVS", "Walgreens", "Starbucks", "McDonald's", "Subway", "Domino's", "Pizza Hut"
            ]
        }
        
        # Fake companies
        self.fake_companies = [
            "Fakester Inc", "Imaginary Corp", "ScamCo", "Phony LLC", "Bogus Enterprises",
            "Fictional Industries", "Fake Solutions", "Mock Technologies", "Pretend Systems",
            "Fabricated Corp", "Counterfeit Industries", "Deceptive Solutions", "Fraudulent Tech",
            "Sham Enterprises", "Hoax Corporation", "Fictitious Industries", "Bogus Solutions",
            "Fake Innovations", "Mock Enterprises", "Pretend Technologies"
        ]
        
        # Real universities by region
        self.real_universities = {
            'us_top': [
                "Harvard University", "Stanford University", "MIT", "Yale University", "Princeton University",
                "Columbia University", "University of Pennsylvania", "Dartmouth College", "Brown University",
                "Cornell University", "University of Chicago", "Northwestern University", "Duke University",
                "Johns Hopkins University", "Vanderbilt University", "Rice University", "Washington University",
                "Emory University", "Georgetown University", "University of Notre Dame"
            ],
            'us_public': [
                "UC Berkeley", "UCLA", "University of Michigan", "University of Virginia", "UNC Chapel Hill",
                "University of Texas at Austin", "University of Wisconsin-Madison", "University of Illinois",
                "University of Washington", "University of Florida", "Georgia Tech", "University of Maryland",
                "Penn State", "Ohio State", "University of Minnesota", "University of Iowa", "Indiana University",
                "Purdue University", "Michigan State", "University of Arizona"
            ],
            'international': [
                "University of Oxford", "University of Cambridge", "Imperial College London", "UCL",
                "London School of Economics", "University of Toronto", "McGill University", "University of British Columbia",
                "University of Melbourne", "University of Sydney", "National University of Singapore",
                "Tsinghua University", "Peking University", "Seoul National University", "KAIST",
                "University of Tokyo", "Kyoto University", "ETH Zurich", "EPFL", "Delft University of Technology"
            ]
        }
        
        # Fake universities
        self.fake_universities = [
            "Fake University", "Diploma Mill College", "Nowhere State University", "Bogus Institute of Technology",
            "Fictional University", "Counterfeit College", "Mock University", "Pretend Institute",
            "Fabricated University", "Deceptive College", "Fraudulent University", "Sham Institute",
            "Hoax University", "Fictitious College", "Bogus University", "Fake Institute of Technology",
            "Mock College", "Pretend University", "Fabricated Institute", "Counterfeit University"
        ]
        
        # Job titles by level and industry
        self.job_titles = {
            'tech': {
                'entry': ["Software Engineer", "Data Analyst", "Product Analyst", "QA Engineer", "DevOps Engineer"],
                'mid': ["Senior Software Engineer", "Data Scientist", "Product Manager", "Engineering Manager", "Technical Lead"],
                'senior': ["Principal Engineer", "Senior Data Scientist", "Senior Product Manager", "Engineering Director", "Architect"],
                'executive': ["CTO", "VP Engineering", "VP Product", "Chief Data Officer", "Chief Technology Officer"]
            },
            'finance': {
                'entry': ["Financial Analyst", "Investment Analyst", "Risk Analyst", "Accountant", "Auditor"],
                'mid': ["Senior Financial Analyst", "Portfolio Manager", "Risk Manager", "Senior Accountant", "Internal Auditor"],
                'senior': ["Finance Manager", "Senior Portfolio Manager", "Senior Risk Manager", "Controller", "Senior Auditor"],
                'executive': ["CFO", "VP Finance", "Chief Risk Officer", "Chief Investment Officer", "Treasurer"]
            },
            'consulting': {
                'entry': ["Business Analyst", "Consultant", "Associate", "Research Analyst", "Strategy Analyst"],
                'mid': ["Senior Consultant", "Manager", "Senior Associate", "Senior Analyst", "Project Manager"],
                'senior': ["Principal", "Senior Manager", "Director", "Senior Principal", "Partner"],
                'executive': ["Managing Director", "Partner", "Senior Partner", "VP Consulting", "Chief Strategy Officer"]
            },
            'healthcare': {
                'entry': ["Medical Assistant", "Nurse", "Pharmacy Technician", "Medical Records Specialist", "Lab Technician"],
                'mid': ["Registered Nurse", "Physician Assistant", "Medical Technologist", "Clinical Specialist", "Healthcare Administrator"],
                'senior': ["Nurse Practitioner", "Clinical Manager", "Healthcare Director", "Medical Director", "Senior Administrator"],
                'executive': ["Chief Medical Officer", "VP Healthcare", "Hospital Administrator", "Medical Director", "Chief Nursing Officer"]
            },
            'retail': {
                'entry': ["Sales Associate", "Cashier", "Stock Clerk", "Customer Service Representative", "Retail Assistant"],
                'mid': ["Store Manager", "Department Manager", "Sales Supervisor", "Customer Service Manager", "Inventory Manager"],
                'senior': ["Regional Manager", "District Manager", "Senior Store Manager", "Operations Manager", "Merchandising Manager"],
                'executive': ["VP Retail", "Chief Operating Officer", "Retail Director", "VP Operations", "Chief Merchandising Officer"]
            }
        }
        
        # Fake job titles
        self.fake_job_titles = [
            "Chief Visionary", "Ninja Developer", "Rockstar Engineer", "Guru Consultant", "Hacker Extraordinaire",
            "Digital Wizard", "Code Ninja", "Tech Rockstar", "Innovation Guru", "Disruption Specialist",
            "Future Architect", "Digital Transformer", "Innovation Catalyst", "Disruption Ninja", "Tech Wizard",
            "Digital Rockstar", "Innovation Ninja", "Future Guru", "Tech Catalyst", "Digital Disruptor"
        ]
        
        # Real addresses
        self.real_addresses = [
            "123 Main St, New York, NY 10001",
            "456 Oak Ave, San Francisco, CA 94102",
            "789 Pine Rd, Chicago, IL 60601",
            "321 Elm St, Boston, MA 02101",
            "654 Maple Dr, Seattle, WA 98101",
            "987 Cedar Ln, Austin, TX 73301",
            "147 Birch Way, Denver, CO 80201",
            "258 Spruce St, Portland, OR 97201",
            "369 Willow Ave, Miami, FL 33101",
            "741 Aspen Rd, Atlanta, GA 30301"
        ]
        
        # Fake addresses
        self.fake_addresses = [
            "1 Infinite Loop, Nowhere, NW 00000",
            "404 Not Found St, Faketown, FT 12345",
            "1234 Unreal Ave, Imaginary City, IC 54321",
            "999 Fake Blvd, Bogus Town, BT 98765",
            "777 Mock Dr, Counterfeit City, CC 11111",
            "555 Pretend Ln, Fabricated Town, FT 22222",
            "333 Sham Way, Deceptive City, DC 33333",
            "111 Hoax Rd, Fraudulent Town, FT 44444",
            "888 Fictitious Ave, Bogus City, BC 55555",
            "666 Fake St, Mock Town, MT 66666"
        ]
        
        # Email domains
        self.email_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]
        
        # Fake email domains
        self.fake_email_domains = ["temp-mail.org", "10minutemail.com", "throwaway.com", "fake-email.com", "bogus-mail.com"]
        
    def generate_real_profile(self, industry='tech'):
        """Generate a realistic authentic profile"""
        # Choose name category
        name_category = random.choice(list(self.real_names.keys()))
        name = random.choice(self.real_names[name_category])
        
        # Generate email
        name_parts = name.lower().split()
        if len(name_parts) >= 2:
            email = f"{name_parts[0]}.{name_parts[1]}@{random.choice(self.email_domains)}"
        else:
            email = f"{name_parts[0]}{random.randint(100, 999)}@{random.choice(self.email_domains)}"
        
        # Generate job history
        job_history = []
        current_year = datetime.now().year
        start_year = random.randint(current_year - 15, current_year - 2)
        
        # Generate 1-4 jobs
        num_jobs = random.randint(1, 4)
        for i in range(num_jobs):
            if i == 0:
                # First job
                level = 'entry'
                duration = random.randint(2, 4)
            elif i == num_jobs - 1:
                # Current job
                level = random.choice(['mid', 'senior'])
                max_duration = max(1, current_year - start_year)
                duration = random.randint(1, max_duration)
            else:
                # Middle jobs
                level = random.choice(['entry', 'mid'])
                duration = random.randint(2, 5)
            
            title = random.choice(self.job_titles[industry][level])
            company = random.choice(self.real_companies[industry])
            
            job = {
                "title": title,
                "company": company,
                "start": start_year,
                "end": start_year + duration
            }
            job_history.append(job)
            start_year += duration
        
        # Generate education
        education = []
        num_degrees = random.randint(1, 2)
        
        for i in range(num_degrees):
            if i == 0:
                # Undergraduate
                degree_type = random.choice(["BSc", "BA", "BS", "BBA"])
                university_category = random.choice(['us_public', 'us_top', 'international'])
                university = random.choice(self.real_universities[university_category])
                major = random.choice([
                    "Computer Science", "Business Administration", "Engineering", "Mathematics",
                    "Economics", "Finance", "Marketing", "Data Science", "Information Systems"
                ])
            else:
                # Graduate (if applicable)
                degree_type = random.choice(["MSc", "MBA", "MA", "MS"])
                university_category = random.choice(['us_top', 'international'])
                university = random.choice(self.real_universities[university_category])
                major = random.choice([
                    "Computer Science", "Business Administration", "Data Science", "Finance",
                    "Engineering Management", "Information Technology", "Statistics"
                ])
            
            education.append({
                "school": university,
                "degree": f"{degree_type} {major}"
            })
        
        return {
            "name": name,
            "email": email,
            "address": random.choice(self.real_addresses),
            "job_history": job_history,
            "education": education,
            "photo_flag": 0,  # Real photo
            "label": 1  # Authentic
        }
    
    def generate_fake_profile(self, fake_type='obvious'):
        """Generate a fake profile with different types of fakery"""
        if fake_type == 'obvious':
            # Obviously fake
            name = random.choice(self.fake_names)
            email = f"user{random.randint(1000, 9999)}@{random.choice(self.fake_email_domains)}"
            address = random.choice(self.fake_addresses)
            
            # Fake job history with suspicious titles
            job_history = []
            current_year = datetime.now().year
            start_year = random.randint(current_year - 10, current_year - 1)
            
            num_jobs = random.randint(1, 3)
            for i in range(num_jobs):
                title = random.choice(self.fake_job_titles)
                company = random.choice(self.fake_companies)
                
                # Create timeline overlaps for obvious fakery
                if i > 0 and random.random() < 0.7:  # 70% chance of overlap
                    start_year = start_year - random.randint(1, 3)
                
                job = {
                    "title": title,
                    "company": company,
                    "start": start_year,
                    "end": start_year + random.randint(1, 4)
                }
                job_history.append(job)
                start_year += random.randint(2, 5)
            
            # Fake education
            education = []
            university = random.choice(self.fake_universities)
            degree_type = random.choice(["Diploma", "Certificate", "Fake Degree"])
            major = random.choice(["Business", "Technology", "Management", "Leadership"])
            
            education.append({
                "school": university,
                "degree": f"{degree_type} in {major}"
            })
            
        elif fake_type == 'subtle':
            # Subtle fake - harder to detect
            name_category = random.choice(list(self.real_names.keys()))
            name = random.choice(self.real_names[name_category])
            
            # Suspicious email pattern
            email = f"{name.lower().replace(' ', '')}{random.randint(100, 999)}@{random.choice(self.email_domains)}"
            address = random.choice(self.real_addresses)
            
            # Real-looking job history but with subtle issues
            job_history = []
            current_year = datetime.now().year
            start_year = random.randint(current_year - 12, current_year - 2)
            
            num_jobs = random.randint(2, 4)
            for i in range(num_jobs):
                if random.random() < 0.3:  # 30% chance of suspicious title
                    title = random.choice(self.fake_job_titles)
                else:
                    title = random.choice(self.job_titles['tech']['mid'])
                
                if random.random() < 0.2:  # 20% chance of fake company
                    company = random.choice(self.fake_companies)
                else:
                    company = random.choice(self.real_companies['tech'])
                
                # Subtle timeline issues
                if i > 0 and random.random() < 0.4:  # 40% chance of small overlap
                    start_year = start_year - random.randint(0, 1)
                
                job = {
                    "title": title,
                    "company": company,
                    "start": start_year,
                    "end": start_year + random.randint(2, 4)
                }
                job_history.append(job)
                start_year += random.randint(2, 4)
            
            # Education with subtle issues
            education = []
            if random.random() < 0.3:  # 30% chance of fake university
                university = random.choice(self.fake_universities)
                degree_type = "Certificate"
            else:
                university = random.choice(self.real_universities['us_public'])
                degree_type = "BSc"
            
            education.append({
                "school": university,
                "degree": f"{degree_type} Computer Science"
            })
        
        elif fake_type == 'ai_generated':
            # AI-generated content with overly positive language
            name = random.choice(self.real_names['western'])
            email = f"{name.lower().replace(' ', '.')}@{random.choice(self.email_domains)}"
            address = random.choice(self.real_addresses)
            
            # Job history with AI-generated language
            job_history = []
            current_year = datetime.now().year
            start_year = random.randint(current_year - 10, current_year - 2)
            
            ai_titles = [
                "Exceptional Senior Software Engineer",
                "Outstanding Product Manager", 
                "Remarkable Data Scientist",
                "Phenomenal Engineering Lead",
                "Fantastic Technical Architect"
            ]
            
            ai_companies = [
                "Outstanding Tech Solutions Inc",
                "Exceptional Digital Innovations",
                "Remarkable Technology Corp",
                "Phenomenal Software Systems",
                "Fantastic Tech Enterprises"
            ]
            
            num_jobs = random.randint(1, 3)
            for i in range(num_jobs):
                title = random.choice(ai_titles)
                company = random.choice(ai_companies)
                
                job = {
                    "title": title,
                    "company": company,
                    "start": start_year,
                    "end": start_year + random.randint(2, 4)
                }
                job_history.append(job)
                start_year += random.randint(2, 4)
            
            # AI-generated education
            ai_universities = [
                "Phenomenal University",
                "Exceptional Institute of Technology",
                "Outstanding University",
                "Remarkable College",
                "Fantastic University"
            ]
            
            education = []
            university = random.choice(ai_universities)
            degree_type = random.choice(["Brilliant", "Exceptional", "Outstanding"])
            major = random.choice(["Computer Science", "Engineering", "Technology"])
            
            education.append({
                "school": university,
                "degree": f"{degree_type} {major} Degree"
            })
        
        return {
            "name": name,
            "email": email,
            "address": address,
            "job_history": job_history,
            "education": education,
            "photo_flag": 1,  # Suspicious photo
            "label": 0  # Fake
        }
    
    def generate_dataset(self, n_real=300, n_fake=300, seed=42):
        """Generate a comprehensive dataset with variety"""
        random.seed(seed)
        np.random.seed(seed)
        
        profiles = []
        
        # Generate real profiles across different industries
        industries = ['tech', 'finance', 'consulting', 'healthcare', 'retail']
        real_per_industry = n_real // len(industries)
        
        for industry in industries:
            for _ in range(real_per_industry):
                profile = self.generate_real_profile(industry)
                profiles.append(profile)
        
        # Add remaining real profiles
        remaining_real = n_real - (real_per_industry * len(industries))
        for _ in range(remaining_real):
            industry = random.choice(industries)
            profile = self.generate_real_profile(industry)
            profiles.append(profile)
        
        # Generate fake profiles with different types
        fake_types = ['obvious', 'subtle', 'ai_generated']
        fake_per_type = n_fake // len(fake_types)
        
        for fake_type in fake_types:
            for _ in range(fake_per_type):
                profile = self.generate_fake_profile(fake_type)
                profiles.append(profile)
        
        # Add remaining fake profiles
        remaining_fake = n_fake - (fake_per_type * len(fake_types))
        for _ in range(remaining_fake):
            fake_type = random.choice(fake_types)
            profile = self.generate_fake_profile(fake_type)
            profiles.append(profile)
        
        # Shuffle the dataset
        random.shuffle(profiles)
        
        return pd.DataFrame(profiles)
    
    def save_dataset(self, df, filename="comprehensive_profiles.json"):
        """Save dataset to JSON file"""
        df.to_json(filename, orient="records", lines=True)
        print(f"✅ Generated {len(df)} profiles")
        print(f"📊 Real profiles: {len(df[df['label'] == 1])}")
        print(f"🚨 Fake profiles: {len(df[df['label'] == 0])}")
        print(f"💾 Saved to: {filename}")
        
        # Print some statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"   Industries covered: {len(self.real_companies)}")
        print(f"   Universities covered: {len(self.real_universities)}")
        print(f"   Job titles: {sum(len(titles) for titles in self.job_titles.values())}")
        print(f"   Name diversity: {sum(len(names) for names in self.real_names.values())}")

if __name__ == "__main__":
    print("🚀 Comprehensive Profile Data Generator")
    print("=" * 50)
    
    generator = ComprehensiveProfileGenerator()
    
    # Generate comprehensive dataset
    df = generator.generate_dataset(n_real=400, n_fake=400)
    
    # Save dataset
    generator.save_dataset(df, "comprehensive_profiles.json")
    
    print(f"\n🎯 Sample Profiles:")
    print("=" * 30)
    
    # Show sample real profile
    real_sample = df[df['label'] == 1].iloc[0]
    print(f"✅ Real Profile Example:")
    print(f"   Name: {real_sample['name']}")
    print(f"   Email: {real_sample['email']}")
    print(f"   Jobs: {len(real_sample['job_history'])} positions")
    print(f"   Education: {len(real_sample['education'])} degrees")
    
    # Show sample fake profile
    fake_sample = df[df['label'] == 0].iloc[0]
    print(f"\n🚨 Fake Profile Example:")
    print(f"   Name: {fake_sample['name']}")
    print(f"   Email: {fake_sample['email']}")
    print(f"   Jobs: {len(fake_sample['job_history'])} positions")
    print(f"   Education: {len(fake_sample['education'])} degrees")
    
    print(f"\n🎉 Dataset generation completed!") 