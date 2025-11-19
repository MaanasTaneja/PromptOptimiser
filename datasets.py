from typing import List



def generate_sentiment_dataset() -> List[dict]:
    """Generate an expanded sentiment classification dataset without mixed labels."""
    return [
        {"input": "I absolutely loved this movie! The acting was stellar.", "output": "positive"},
        {"input": "This product is terrible. Complete waste of money.", "output": "negative"},
        {"input": "The service was okay, nothing special.", "output": "neutral"},
        {"input": "Best purchase I've ever made!", "output": "positive"},
        {"input": "Horrible experience. Would not recommend.", "output": "negative"},
        {"input": "It's fine, does what it's supposed to do.", "output": "neutral"},
        
        {"input": "The food was amazing, and the ambiance felt perfect.", "output": "positive"},
        {"input": "I’m not impressed, but it’s not awful either.", "output": "neutral"},
        {"input": "Absolutely fantastic! Will be coming back for sure.", "output": "positive"},
        {"input": "The update made everything slower. Very disappointing.", "output": "negative"},
        
        {"input": "I guess it works... most of the time.", "output": "neutral"},
        {"input": "Wow, just wow. This exceeded every expectation!", "output": "positive"},
        {"input": "I can’t believe I paid money for this junk.", "output": "negative"},
        {"input": "Not great, not terrible, just average.", "output": "neutral"},
        
        {"input": "I'm blown away by how good this turned out!", "output": "positive"},
        {"input": "This made my day. Absolutely loved it!", "output": "positive"},
        {"input": "Meh. I don’t really care for it.", "output": "neutral"},
        {"input": "Terrible product. It broke on the first day.", "output": "negative"},
        {"input": "Pretty bad overall. Would not buy again.", "output": "negative"},
        {"input": "Surprisingly good quality for the price.", "output": "positive"},
    ]



def generate_qa_dataset() -> List[dict]:
    """Generate a larger factual QA dataset."""
    return [
        {"input": "What is the capital of France?", "output": "paris"},
        {"input": "Who wrote Romeo and Juliet?", "output": "shakespeare"},
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is the largest planet in the solar system?", "output": "jupiter"},
        {"input": "In what year did World War 2 end?", "output": "1945"},
        {"input": "What is the chemical symbol for water?", "output": "h2o"},
        {"input": "Who painted the Mona Lisa?", "output": "da vinci"},
        {"input": "What gas do plants absorb for photosynthesis?", "output": "carbon dioxide"},
        {"input": "What is the capital of Japan?", "output": "tokyo"},
        {"input": "How many continents are there?", "output": "7"},
        {"input": "Who discovered gravity according to legend?", "output": "newton"},
        {"input": "What is the tallest mountain on Earth?", "output": "everest"},
        {"input": "What is the hardest natural substance?", "output": "diamond"},
        {"input": "What is the square root of 64?", "output": "8"},
        {"input": "Which ocean is the largest?", "output": "pacific"},
        {"input": "Who is known as the father of computers?", "output": "charles babbage"},
        {"input": "What planet is known as the Red Planet?", "output": "mars"},
        {"input": "Which metal is liquid at room temperature?", "output": "mercury"},
        {"input": "What is the currency of the United Kingdom?", "output": "pound"},
        {"input": "What organ pumps blood through the body?", "output": "heart"},
    ]



def generate_summarization_dataset() -> List[dict]:
    """Generate a larger text summarization dataset."""
    return [
        {
            "input": "The quick brown fox jumps over the lazy dog. This sentence is used to test typing and fonts.",
            "output": "pangram about fox"
        },
        {
            "input": "Machine learning is a subset of artificial intelligence that enables systems to learn patterns from data and improve over time.",
            "output": "ml learns from data"
        },
        {
            "input": "Python is a high-level programming language favored for its clean syntax and broad ecosystem of libraries.",
            "output": "python is clean and versatile"
        },
        {
            "input": "The Amazon rainforest, often called the planet's lungs, plays a crucial role in absorbing carbon dioxide and producing oxygen.",
            "output": "amazon rainforest absorbs co2"
        },
        {
            "input": "Photosynthesis allows plants to convert light into chemical energy, sustaining both their growth and the global food chain.",
            "output": "photosynthesis creates plant energy"
        },
        {
            "input": "The Great Wall of China was built over centuries to protect ancient Chinese states from invasions along the northern borders.",
            "output": "great wall built for defense"
        },
        {
            "input": "The Internet transformed global communication by enabling instant information exchange across vast distances.",
            "output": "internet enables instant communication"
        },
        {
            "input": "Electric vehicles are becoming increasingly popular due to their reduced emissions and lower long-term maintenance costs.",
            "output": "ev popularity due to low emissions"
        },
        {
            "input": "Marie Curie conducted pioneering research on radioactivity, becoming the first woman to win a Nobel Prize.",
            "output": "marie curie pioneered radioactivity"
        },
        {
            "input": "Vaccines work by training the immune system to recognize harmful pathogens and mount a quick response upon exposure.",
            "output": "vaccines train immunity"
        },
        {
            "input": "Cloud computing allows businesses to scale their applications easily without needing to manage physical servers.",
            "output": "cloud enables scalable apps"
        },
        {
            "input": "Mount Everest is the tallest mountain on Earth, attracting climbers from around the world despite its dangerous conditions.",
            "output": "mount everest world's tallest"
        },
        {
            "input": "The Renaissance was a cultural movement that emphasized art, science, and humanism, reshaping European history.",
            "output": "renaissance revived art and science"
        },
        {
            "input": "Renewable energy sources like wind and solar are becoming essential in the fight against climate change.",
            "output": "renewables help fight climate change"
        },
        {
            "input": "Artificial intelligence has enabled major breakthroughs in language understanding, robotics, and decision-making systems.",
            "output": "ai enables major breakthroughs"
        },
        {
            "input": "The Mars Rover missions have provided critical information about the planet's climate, geology, and potential for past life.",
            "output": "mars rovers reveal planetary data"
        },
        {
            "input": "Global supply chains were significantly disrupted during the pandemic due to factory shutdowns and transportation delays.",
            "output": "pandemic disrupted global supply chains"
        },
        {
            "input": "Cryptocurrencies rely on decentralized blockchain networks to verify transactions without the need for central authorities.",
            "output": "crypto uses decentralized blockchain"
        },
        {
            "input": "Sleep is vital for cognitive function, emotional health, and physical recovery, affecting nearly every aspect of well-being.",
            "output": "sleep essential for health"
        },
        {
            "input": "The printing press revolutionized information sharing by enabling mass production of books at low cost.",
            "output": "printing press enabled mass books"
        },
    ]


def generate_reasoning_dataset() -> list[dict]:
    """Generate complex reasoning dataset requiring nuanced evaluation."""
    return [
        {
            "input": "Why might renewable energy sources be better than fossil fuels for long-term sustainability?",
            "output": "renewables reduce emissions, are sustainable long-term, and have lower environmental impact"
        },
        {
            "input": "What are the pros and cons of remote work?",
            "output": "pros include flexibility and cost savings; cons include isolation and communication challenges"
        },
        {
            "input": "Explain artificial intelligence in simple terms.",
            "output": "ai means computers doing tasks requiring human-like reasoning learning and problem solving"
        },
        {
            "input": "Why do some people prefer coffee over tea?",
            "output": "due to caffeine strength taste preferences cultural habits and personal caffeine tolerance"
        },
        {
            "input": "What's the relationship between education and career success?",
            "output": "education builds skills and credentials that improve job opportunities and earning potential"
        },
        {
            "input": "Why do governments impose taxes?",
            "output": "to fund public services redistribute resources and influence economic behavior"
        },
        {
            "input": "Why do humans form social groups?",
            "output": "groups provide safety cooperation shared resources and emotional belonging"
        },
        {
            "input": "Why might exercise improve mental health?",
            "output": "exercise reduces stress increases endorphins and supports better sleep and mood regulation"
        },
        {
            "input": "Why is critical thinking important?",
            "output": "it helps evaluate information make logical decisions and avoid cognitive biases"
        },
        {
            "input": "Why do companies invest in research and development?",
            "output": "to innovate improve products stay competitive and create long-term growth"
        },
        {
            "input": "Why can inflation hurt consumers?",
            "output": "inflation reduces purchasing power making goods and services more expensive"
        },
        {
            "input": "Why do people vote in elections?",
            "output": "to influence government policy choose leaders and participate in democracy"
        },
        {
            "input": "Why might collaboration lead to better ideas than individual work?",
            "output": "collaboration combines diverse perspectives sparks creativity and reduces blind spots"
        },
        {
            "input": "Why is sleep essential for learning?",
            "output": "sleep strengthens memory consolidates new knowledge and restores cognitive function"
        },
        {
            "input": "Why do many species migrate?",
            "output": "migration helps find food better climate suitable breeding grounds and survival advantages"
        },
        {
            "input": "Why is financial literacy important?",
            "output": "it helps people manage money avoid debt build savings and make informed decisions"
        },
        {
            "input": "Why do humans create art?",
            "output": "art expresses emotions communicates ideas and strengthens cultural identity"
        },
        {
            "input": "Why are scientific experiments repeated multiple times?",
            "output": "repetition improves reliability reduces error and confirms consistent results"
        },
        {
            "input": "Why do businesses create budgets?",
            "output": "budgets control spending set priorities and help predict future financial needs"
        },
        {
            "input": "Why is empathy important in relationships?",
            "output": "empathy improves understanding builds trust and strengthens emotional connection"
        },
    ]



def generate_nli_dataset() -> list[dict]:
    """Generate a 20-sample Natural Language Inference dataset requiring semantic understanding."""
    return [
        # Entailment (1–7)
        {
            "input": "Premise: All dogs are animals. Hypothesis: Some animals are dogs.",
            "output": "entailment"
        },
        {
            "input": "Premise: The cat sat on the mat. Hypothesis: The mat was under the cat.",
            "output": "entailment"
        },
        {
            "input": "Premise: The event starts at 6pm. Hypothesis: The event starts in the evening.",
            "output": "entailment"
        },
        {
            "input": "Premise: John bought a red car yesterday. Hypothesis: John owns a car.",
            "output": "entailment"
        },
        {
            "input": "Premise: Every student in the class submitted their assignment. Hypothesis: Sarah, who is in the class, submitted her assignment.",
            "output": "entailment"
        },
        {
            "input": "Premise: The glass fell off the table and shattered. Hypothesis: The glass broke.",
            "output": "entailment"
        },
        {
            "input": "Premise: The restaurant closes at 10pm. Hypothesis: The restaurant is not open at 11pm.",
            "output": "entailment"
        },

        # Contradiction (8–14)
        {
            "input": "Premise: John loves pizza. Hypothesis: John dislikes pizza.",
            "output": "contradiction"
        },
        {
            "input": "Premise: The book is on the table. Hypothesis: The book is not on the table.",
            "output": "contradiction"
        },
        {
            "input": "Premise: She arrived at the office at 9am. Hypothesis: She arrived at the office at noon.",
            "output": "contradiction"
        },
        {
            "input": "Premise: All of the cookies were eaten. Hypothesis: There are still some cookies left.",
            "output": "contradiction"
        },
        {
            "input": "Premise: The movie was two hours long. Hypothesis: The movie lasted only thirty minutes.",
            "output": "contradiction"
        },
        {
            "input": "Premise: He is an only child. Hypothesis: He has two brothers.",
            "output": "contradiction"
        },
        {
            "input": "Premise: The temperature is below freezing. Hypothesis: The temperature is warm enough to melt ice.",
            "output": "contradiction"
        },

        # Neutral (15–20)
        {
            "input": "Premise: She speaks English fluently. Hypothesis: She is from England.",
            "output": "neutral"
        },
        {
            "input": "Premise: Tom bought a new laptop. Hypothesis: Tom bought the laptop on sale.",
            "output": "neutral"
        },
        {
            "input": "Premise: The man is reading a book on the train. Hypothesis: The man is reading a mystery novel.",
            "output": "neutral"
        },
        {
            "input": "Premise: The company announced a new product today. Hypothesis: The product will be successful.",
            "output": "neutral"
        },
        {
            "input": "Premise: A child is playing with a ball in the park. Hypothesis: The child is five years old.",
            "output": "neutral"
        },
        {
            "input": "Premise: The team won their game yesterday. Hypothesis: The team celebrated at a restaurant afterward.",
            "output": "neutral"
        }
    ]
