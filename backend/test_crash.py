import traceback
with open("crash_log.txt", "w") as f:
    try:
        import main
        f.write("SUCCESS\n")
    except Exception as e:
        f.write(f"FAILED: {str(e)}\n")
        f.write(traceback.format_exc())
