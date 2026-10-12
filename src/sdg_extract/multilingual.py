from __future__ import annotations

from typing import Dict, List


SUPPORTED_LANGUAGE_CODES = ["eng", "spa", "por", "vie", "fra", "zho", "cmn", "deu", "ara", "ind", "npi", "jpn"]

COUNTRY_ALIASES: Dict[str, str] = {
    "china": "China", "chn": "China", "中国": "China", "中华人民共和国": "China", "chine": "China",
    "united states": "United States", "usa": "United States", "u.s.": "United States", "美国": "United States", "estados unidos": "United States", "états-unis": "United States",
    "canada": "Canada", "canadá": "Canada",
    "mexico": "Mexico", "méxico": "Mexico",
    "brazil": "Brazil", "brasil": "Brazil",
    "vietnam": "Vietnam", "việt nam": "Vietnam",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "英国": "United Kingdom", "royaume-uni": "United Kingdom", "reino unido": "United Kingdom",
    "egypt": "Egypt", "مصر": "Egypt", "egipto": "Egypt",
    "germany": "Germany", "deutschland": "Germany", "allemagne": "Germany", "alemania": "Germany", "德国": "Germany",
    "indonesia": "Indonesia",
    "nepal": "Nepal", "नेपाल": "Nepal",
    "india": "India", "भारत": "India", "印度": "India",
    "japan": "Japan", "japón": "Japan", "japon": "Japan", "日本": "Japan",
    "mozambique": "Mozambique", "moçambique": "Mozambique",
    "philippines": "Philippines", "filipinas": "Philippines",
    "uganda": "Uganda",
    "kenya": "Kenya",
    "madagascar": "Madagascar",
    "south africa": "South Africa", "sudáfrica": "South Africa", "afrique du sud": "South Africa",
    "ethiopia": "Ethiopia", "ethiopie": "Ethiopia", "etiopía": "Ethiopia",
}

CLIMATE_TERMS: List[str] = [
    "climate", "greenhouse gas", "emissions", "mitigation", "adaptation", "resilience", "carbon", "net zero", "ndc", "unfccc", "paris agreement", "disaster risk", "early warning", "climate finance",
    "气候", "温室气体", "排放", "减排", "碳", "适应", "韧性", "灾害风险", "预警", "气候融资",
    "clima", "gases de efecto invernadero", "emisiones", "mitigación", "adaptación", "resiliencia", "carbono", "cero neto", "riesgo de desastres", "alerta temprana", "financiación climática",
    "gases com efeito de estufa", "emissões", "mitigação", "adaptação", "resiliência", "carbono", "zero líquido", "risco de desastres", "alerta precoce", "financiamento climático",
    "climat", "gaz à effet de serre", "émissions", "atténuation", "adaptation", "résilience", "carbone", "zéro émission nette", "risque de catastrophe", "alerte précoce", "financement climatique",
    "klima", "treibhausgas", "emissionen", "minderung", "anpassung", "resilienz", "kohlenstoff", "klimaneutral", "katastrophenrisiko", "frühwarnung", "klimafinanzierung",
    "khí hậu", "khí nhà kính", "phát thải", "giảm nhẹ", "thích ứng", "chống chịu", "carbon", "phát thải ròng bằng không", "rủi ro thiên tai", "cảnh báo sớm", "tài chính khí hậu",
    "المناخ", "غازات الدفيئة", "انبعاثات", "التخفيف", "التكيف", "المرونة", "الكربون", "صافي صفر", "مخاطر الكوارث", "الإنذار المبكر", "تمويل المناخ",
    "iklim", "gas rumah kaca", "emisi", "mitigasi", "adaptasi", "ketahanan", "karbon", "nol bersih", "risiko bencana", "peringatan dini", "pendanaan iklim",
    "जलवायु", "हरितगृह ग्यास", "उत्सर्जन", "न्यूनीकरण", "अनुकूलन", "उत्थानशीलता", "कार्बन", "विपद् जोखिम", "पूर्व चेतावनी", "जलवायु वित्त",
    "気候", "温室効果ガス", "排出", "緩和", "適応", "レジリエンス", "炭素", "ネットゼロ", "災害リスク", "早期警戒", "気候資金",
]

TARGET_TERMS: Dict[str, List[str]] = {
    "13.1": ["adaptation", "adaptive", "resilience", "disaster risk", "extreme weather", "适应", "韧性", "灾害风险", "极端天气", "adaptación", "resiliencia", "riesgo de desastres", "adaptação", "risco de desastres", "adaptation", "résilience", "risque de catastrophe", "anpassung", "resilienz", "katastrophenrisiko", "thích ứng", "chống chịu", "rủi ro thiên tai", "التكيف", "المرونة", "مخاطر الكوارث", "adaptasi", "ketahanan", "risiko bencana", "अनुकूलन", "उत्थानशीलता", "विपद् जोखिम", "適応", "レジリエンス", "災害リスク"],
    "13.2": ["climate policy", "climate strategy", "climate plan", "policy", "strategy", "planning", "law", "regulation", "ndc", "气候政策", "战略", "规划", "法律", "法规", "国家自主贡献", "política climática", "estrategia climática", "plan climático", "ley", "reglamento", "política climática", "planejamento", "lei", "regulamento", "politique climatique", "stratégie climatique", "planification", "loi", "règlement", "klimapolitik", "klimastrategie", "planung", "gesetz", "verordnung", "chính sách khí hậu", "chiến lược", "kế hoạch", "luật", "quy định", "سياسة المناخ", "استراتيجية", "تخطيط", "قانون", "لائحة", "kebijakan iklim", "strategi", "perencanaan", "undang-undang", "peraturan", "जलवायु नीति", "रणनीति", "योजना", "कानून", "नियमन", "気候政策", "戦略", "計画", "法律", "規制"],
    "13.3": ["climate education", "awareness", "training", "capacity", "early warning", "气候教育", "意识", "培训", "能力建设", "预警", "educación climática", "concienciación", "capacitación", "alerta temprana", "educação climática", "conscientização", "capacitação", "alerte précoce", "éducation climatique", "sensibilisation", "formation", "capacité", "klimabildung", "bewusstsein", "schulung", "frühwarnung", "giáo dục khí hậu", "nhận thức", "đào tạo", "cảnh báo sớm", "التثقيف المناخي", "الوعي", "التدريب", "الإنذار المبكر", "pendidikan iklim", "kesadaran", "pelatihan", "peringatan dini", "जलवायु शिक्षा", "चेतना", "तालिम", "पूर्व चेतावनी", "気候教育", "意識", "研修", "早期警戒"],
    "13.a": ["climate finance", "green climate fund", "$100 billion", "100 billion", "气候融资", "绿色气候基金", "千亿美元", "financiación climática", "fondo verde para el clima", "financiamento climático", "fundo verde para o clima", "financement climatique", "fonds vert pour le climat", "klimafinanzierung", "grüner klimafonds", "tài chính khí hậu", "quỹ khí hậu xanh", "تمويل المناخ", "الصندوق الأخضر للمناخ", "pendanaan iklim", "green climate fund", "जलवायु वित्त", "हरित जलवायु कोष", "気候資金", "緑の気候基金"],
    "13.b": ["least developed", "small island", "ldc", "sids", "最不发达国家", "小岛屿发展中国家", "países menos adelantados", "pequeños estados insulares", "países menos desenvolvidos", "pequenos estados insulares", "pays les moins avancés", "petits états insulaires", "am wenigsten entwickelte länder", "kleine inselstaaten", "các nước kém phát triển", "các quốc đảo nhỏ", "أقل البلدان نموا", "الدول الجزرية الصغيرة", "negara kurang berkembang", "negara kepulauan kecil", "अतिकम विकसित देश", "साना टापु विकासशील राज्य", "後発開発途上国", "小島嶼開発途上国"],
}

TARGET_13_1_ACTION_TERMS: List[str] = [
    "adaptation", "adaptive", "resilience", "preparedness", "disaster risk reduction",
    "适应", "韧性", "防灾减灾", "adaptación", "resiliencia", "preparación",
    "adaptação", "resiliência", "preparação", "adaptation", "résilience", "préparation",
    "anpassung", "resilienz", "vorsorge", "thích ứng", "chống chịu", "chuẩn bị",
    "التكيف", "المرونة", "التأهب", "adaptasi", "ketahanan", "kesiapsiagaan",
    "अनुकूलन", "उत्थानशीलता", "तयारी", "適応", "レジリエンス", "備え",
    "climate adaptation", "adaptive capacity", "climate resilience", "disaster risk reduction", "climate preparedness",
    "气候适应", "适应能力", "气候韧性", "灾害风险降低", "防灾减灾",
    "adaptación climática", "capacidad de adaptación", "resiliencia climática", "reducción del riesgo de desastres",
    "adaptação climática", "capacidade adaptativa", "resiliência climática", "redução do risco de desastres",
    "adaptation climatique", "capacité d’adaptation", "résilience climatique", "réduction des risques de catastrophe",
    "klimaanpassung", "anpassungsfähigkeit", "klimaresilienz", "katastrophenrisikominderung",
    "thích ứng khí hậu", "năng lực thích ứng", "khả năng chống chịu khí hậu", "giảm thiểu rủi ro thiên tai",
    "التكيف مع المناخ", "القدرة على التكيف", "القدرة على الصمود أمام المناخ", "الحد من مخاطر الكوارث",
    "adaptasi iklim", "kapasitas adaptif", "ketahanan iklim", "pengurangan risiko bencana",
    "जलवायु अनुकूलन", "अनुकूलन क्षमता", "जलवायु उत्थानशीलता", "विपद् जोखिम न्यूनीकरण",
    "気候適応", "適応能力", "気候レジリエンス", "災害リスク軽減",
]

TARGET_13_2_INTEGRATION_TERMS: List[str] = [
    "climate policy", "climate strategy", "climate plan", "climate law", "climate regulation",
    "national climate", "nationally determined contribution", "ndc", "integrate climate", "mainstream climate",
    "气候政策", "气候战略", "气候规划", "气候法律", "气候法规", "国家自主贡献", "将气候", "纳入气候",
    "política climática", "estrategia climática", "plan climático", "ley climática", "contribución determinada a nivel nacional", "integrar el clima",
    "política climática", "estratégia climática", "plano climático", "lei climática", "contribuição nacionalmente determinada", "integrar o clima",
    "politique climatique", "stratégie climatique", "plan climatique", "loi climat", "contribution déterminée au niveau national", "intégrer le climat",
    "klimapolitik", "klimastrategie", "klimaplan", "klimagesetz", "national festgelegter beitrag", "klima integrieren",
    "chính sách khí hậu", "chiến lược khí hậu", "kế hoạch khí hậu", "luật khí hậu", "đóng góp do quốc gia tự quyết định", "lồng ghép khí hậu",
    "سياسة المناخ", "استراتيجية المناخ", "خطة المناخ", "قانون المناخ", "المساهمة المحددة وطنيا", "إدماج المناخ",
    "kebijakan iklim", "strategi iklim", "rencana iklim", "undang-undang iklim", "kontribusi yang ditetapkan secara nasional", "mengintegrasikan iklim",
    "जलवायु नीति", "जलवायु रणनीति", "जलवायु योजना", "जलवायु कानून", "राष्ट्रिय निर्धारित योगदान", "जलवायुलाई मूलप्रवाहीकरण",
    "気候政策", "気候戦略", "気候計画", "気候法", "国が決定する貢献", "気候変動を組み込",
]

TARGET_13_B_CAPACITY_TERMS: List[str] = [
    "planning capacity", "management capacity", "capacity building", "technical assistance", "institutional capacity",
    "规划能力", "管理能力", "能力建设", "技术援助", "机构能力",
    "capacidad de planificación", "capacidad de gestión", "desarrollo de capacidades", "asistencia técnica",
    "capacidade de planejamento", "capacidade de gestão", "capacitação", "assistência técnica",
    "capacité de planification", "capacité de gestion", "renforcement des capacités", "assistance technique",
    "planungskapazität", "managementkapazität", "kapazitätsaufbau", "technische unterstützung",
    "năng lực lập kế hoạch", "năng lực quản lý", "xây dựng năng lực", "hỗ trợ kỹ thuật",
    "قدرة التخطيط", "قدرة الإدارة", "بناء القدرات", "المساعدة التقنية",
    "kapasitas perencanaan", "kapasitas pengelolaan", "pengembangan kapasitas", "bantuan teknis",
    "योजना क्षमता", "व्यवस्थापन क्षमता", "क्षमता विकास", "प्राविधिक सहायता",
    "計画能力", "管理能力", "能力構築", "技術支援",
]

# Candidate ranking and deterministic validation use action-specific gates.
TARGET_TERMS["13.1"] = TARGET_13_1_ACTION_TERMS
TARGET_TERMS["13.2"] = TARGET_13_2_INTEGRATION_TERMS


DATABASE_TERMS = CLIMATE_TERMS + [
    "policy", "policies", "database", "record", "sdg", "target", "政策", "数据库", "记录",
    "política", "políticas", "base de datos", "politique", "politiques", "base de données",
    "politik", "datenbank", "chính sách", "cơ sở dữ liệu", "سياسة", "سياسات", "قاعدة بيانات",
    "kebijakan", "basis data", "नीति", "नीतिहरू", "डाटाबेस", "気候政策", "データベース",
]

AGGREGATE_TERMS = [
    "总结", "全部", "所有", "多少", "数量", "概括", "汇总", "list all", "summarize", "all policies", "how many", "count",
    "resumir", "resumen", "todas las políticas", "cuántas", "resumo", "todas as políticas", "quantas",
    "résumer", "toutes les politiques", "combien", "zusammenfassen", "alle maßnahmen", "wie viele",
    "tóm tắt", "tất cả", "bao nhiêu", "لخص", "ملخص", "جميع السياسات", "كم عدد",
    "rangkum", "semua kebijakan", "berapa", "सारांश", "सबै नीति", "कति", "要約", "すべての政策", "何件",
]

COMPARE_TERMS = [
    "比较", "对比", "区别", "差异", "compare", "difference", "versus", " vs ",
    "comparar", "diferencia", "comparação", "diferença", "comparer", "différence",
    "vergleichen", "unterschied", "so sánh", "khác biệt", "مقارنة", "الفرق",
    "bandingkan", "perbedaan", "तुलना", "भिन्नता", "比較", "違い",
]

QUERY_EXPANSIONS = {
    "气候适应": "climate adaptation resilience", "气候韧性": "climate resilience adaptation", "减排": "emissions reduction mitigation", "温室气体": "greenhouse gas emissions", "甲烷": "methane emissions", "气候融资": "climate finance Green Climate Fund", "气候教育": "climate education awareness capacity", "灾害风险": "disaster risk resilience", "碳中和": "carbon neutrality net zero",
    "adaptación climática": "climate adaptation resilience", "reducción de emisiones": "emissions reduction mitigation", "financiación climática": "climate finance", "educación climática": "climate education awareness",
    "adaptação climática": "climate adaptation resilience", "redução de emissões": "emissions reduction mitigation", "financiamento climático": "climate finance",
    "adaptation climatique": "climate adaptation resilience", "réduction des émissions": "emissions reduction mitigation", "financement climatique": "climate finance",
    "klimaanpassung": "climate adaptation resilience", "emissionsminderung": "emissions reduction mitigation", "klimafinanzierung": "climate finance",
    "thích ứng khí hậu": "climate adaptation resilience", "giảm phát thải": "emissions reduction mitigation", "tài chính khí hậu": "climate finance",
    "التكيف مع المناخ": "climate adaptation resilience", "خفض الانبعاثات": "emissions reduction mitigation", "تمويل المناخ": "climate finance",
    "adaptasi iklim": "climate adaptation resilience", "pengurangan emisi": "emissions reduction mitigation", "pendanaan iklim": "climate finance",
    "जलवायु अनुकूलन": "climate adaptation resilience", "उत्सर्जन न्यूनीकरण": "emissions reduction mitigation", "जलवायु वित्त": "climate finance",
    "気候適応": "climate adaptation resilience", "排出削減": "emissions reduction mitigation", "気候資金": "climate finance",
}

QUERY_STOP_PHRASES = [
    "请问", "请帮我", "帮我", "总结", "全部", "所有", "哪些", "什么", "数据库", "记录",
    "what", "which", "about", "summarize", "list all", "policy", "policies", "database", "records",
    "resumir", "resumen", "política", "políticas", "base de datos", "résumer", "politique", "politiques",
    "zusammenfassen", "politik", "tóm tắt", "chính sách", "لخص", "سياسة", "سياسات",
    "rangkum", "kebijakan", "सारांश", "नीति", "要約", "政策",
]


def expand_multilingual_query(query: str) -> str:
    value = query.lower()
    expanded = [english for phrase, english in QUERY_EXPANSIONS.items() if phrase in value]
    return " ".join(dict.fromkeys(expanded)) if expanded else query
