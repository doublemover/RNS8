TEST_CASE("public status strings cover every ABI status") {
  struct Case {
    rns8_status status;
    const char* text;
  };
  const Case cases[] = {
      {RNS8_SUCCESS, "success"},
      {RNS8_INVALID_ARGUMENT, "invalid argument"},
      {RNS8_UNSUPPORTED_OS, "unsupported operating system"},
      {RNS8_UNSUPPORTED_ARCH, "unsupported architecture"},
      {RNS8_UNSUPPORTED_BACKEND, "unsupported backend"},
      {RNS8_RANGE_ERROR, "range error"},
      {RNS8_ACCUMULATION_OVERFLOW_RISK, "accumulation overflow risk"},
      {RNS8_WORKSPACE_TOO_SMALL, "workspace too small"},
      {RNS8_BACKEND_FAILURE, "backend failure"},
      {RNS8_VERIFICATION_FAILED, "verification failed"},
      {RNS8_INTERNAL_ERROR, "internal error"},
  };

  for (const Case& item : cases) {
    CHECK(std::string(rns8_status_string(item.status)) == item.text);
  }
  CHECK(std::string(rns8_status_string(static_cast<rns8_status>(0x7fff))) == "unknown status");
}
