from supabase import create_client

# =========================
# SUPABASE CONFIG
# =========================
SUPABASE_URL = "https://oydklhdfhcjwaggxfeak.supabase.co"
SUPABASE_KEY = "sb_secret_riiZofPfWYVoAEH0zhterg_ZfgPo-71"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# TEST FUNCTION
# =========================
def pest_lookup(pest_name):
    try:
        print(f"🔍 Searching for pest: {pest_name}")

        response = (
            supabase
            .table("pestguard")
            .select("pest_name,effects,natural_solution,chemical_solution")
            .ilike("pest_name", pest_name)
            .limit(1)
            .execute()
        )

        if response.data:
            row = response.data[0]
            print("✅ FOUND")
            print("Pest Name :", row["pest_name"])
            print("Effect    :", row["effects"])
            print("Natural   :", row["natural_solution"])
            print("Pesticide :", row["chemical_solution"])
        else:
            print("❌ No matching pest found")

    except Exception as e:
        print("🔥 ERROR:", e)


# =========================
# RUN TEST
# =========================
if __name__ == "__main__":
    pest_lookup("moth")
