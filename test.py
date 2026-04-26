from supabase import create_client
import config
import json

# -----------------------
# Supabase init
# -----------------------
supabase = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_KEY
)


def get_pest_info_lang(pest_name, lang="en"):
    """
    Fetch pest info based on language
    """

    print(f"\n🔍 Fetching: {pest_name} | Language: {lang}")

    try:
        # 1️⃣ Get pest ID from base table
        pest_res = (
            supabase
            .table("pestguard")
            .select("id,pest_name")
            .eq("pest_name", pest_name)
            .limit(1)
            .execute()
        )

        if not pest_res.data:
            print("❌ Pest not found")
            return

        pest = pest_res.data[0]
        pest_id = pest["id"]

        print(f"✅ Found pest ID: {pest_id}")

        # 2️⃣ Fetch translation
        trans_res = (
            supabase
            .table("pestguard_translations")
            .select("effects,natural_solution,chemical_solution,lang")
            .eq("pest_id", pest_id)
            .eq("lang", lang)
            .limit(1)
            .execute()
        )

        # 3️⃣ Fallback to English
        if not trans_res.data:
            print("⚠️ Translation not found → fallback to EN")

            trans_res = (
                supabase
                .table("pestguard_translations")
                .select("effects,natural_solution,chemical_solution,lang")
                .eq("pest_id", pest_id)
                .eq("lang", "en")
                .limit(1)
                .execute()
            )

        if trans_res.data:
            row = trans_res.data[0]

            result = {
                "effects": row["effects"],
                "natural_solution": row["natural_solution"],
                "chemical_solution": row["chemical_solution"],
                "lang": row["lang"]
            }

            print("\n✅ RESULT:")
            print(json.dumps(result, indent=4, ensure_ascii=False))

            return result

        else:
            print("❌ No translation data found")

    except Exception as e:
        print("❌ ERROR:", e)


# -----------------------
# RUN TESTS
# -----------------------
if __name__ == "__main__":

    # Test English
    get_pest_info_lang("aphids", "en")

    # Test Hindi
    get_pest_info_lang("aphids", "hi")

    # Test fallback
    get_pest_info_lang("aphids", "fr")