import ast, asyncio, hashlib, json, logging, pathlib, re, subprocess, unittest
from types import SimpleNamespace as NS

BASE = pathlib.Path(__file__).resolve().parents[1]
SOURCE = BASE / "custom_components/llmvision/providers.py"

class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.elapsed = 0.0
        self.calls = []
        self.providers = {}
        self.fallback = None
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Request")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        self.Error = type("ServiceValidationError", (Exception,), {})
        env = dict(time=NS(monotonic=lambda: self.elapsed), _LOGGER=logging.getLogger("offline-timing-test"),
            async_get_clientsession=lambda hass: None, ServiceValidationError=self.Error,
            DOMAIN="llmvision", CONF_TEMPERATURE="temperature", CONF_TOP_P="top_p",
            ERROR_NO_IMAGE_INPUT="no_image", ERROR_GROQ_MULTIPLE_IMAGES="groq_images", ERROR_NOT_CONFIGURED="missing_provider",
            re=re, json=json, ProviderFactory=NS(create=lambda **kw: self.providers[kw["provider_name"]]))
        exec(compile(module, str(SOURCE), "exec"), env)
        self.Request = env["Request"]
        self.Request.get_provider = staticmethod(lambda hass, entry: entry)
        self.Request.get_default_model = staticmethod(lambda entry: "offline-model")
        self.hass = NS(data={"llmvision": {"primary": {}, "secondary": {}}},
            config_entries=NS(async_entries=lambda domain: [NS(data={"provider": "Settings", "fallback_provider": self.fallback})]))

    def request(self):
        req = self.Request(self.hass, message="offline", max_tokens=100, temperature=0.3)
        req.base64_images = ["offline-placeholder"]
        return req

    def call_data(self, title=True):
        return NS(provider="primary", model="offline-model", generate_title=title,
            response_format="text", title_field=None, memory=NS(title_prompt="Title: "), message="offline")

    def provider(self, name, vision_s, title_s=0.0, vision_error=None, title_error=None):
        async def vision(call):
            self.calls.append((name, "vision"))
            self.elapsed += vision_s
            if vision_error is not None:
                raise vision_error
            return "scene " + name
        async def title(call):
            self.calls.append((name, "title"))
            self.elapsed += title_s
            if title_error is not None:
                raise title_error
            return "title " + name
        self.providers[name] = NS(vision_request=vision, title_request=title, supports_structured_output=lambda: False)

    async def test_main_and_title(self):
        self.provider("primary", .3, .1)
        req = self.request()
        result = await req.call(self.call_data())
        self.assertEqual(result, {"response_text": "scene primary", "title": "title primary"})
        self.assertAlmostEqual(req.timings["vision_request_s"], .3)
        self.assertAlmostEqual(req.timings["title_request_s"], .1)

    async def test_title_off(self):
        self.provider("primary", .3, .1)
        req = self.request()
        result = await req.call(self.call_data(False))
        self.assertEqual(result, {"response_text": "scene primary"})
        self.assertEqual(req.timings["title_request_s"], 0)
        self.assertEqual(self.calls, [("primary", "vision")])

    async def test_primary_fallback_not_double_counted(self):
        self.fallback = "secondary"
        self.provider("primary", .2, vision_error=RuntimeError("synthetic primary failure"))
        self.provider("secondary", .3, .1)
        req = self.request()
        result = await req.call(self.call_data())
        self.assertEqual(result["response_text"], "scene secondary")
        self.assertAlmostEqual(req.timings["vision_request_s"], .5)
        self.assertAlmostEqual(req.timings["title_request_s"], .1)
        self.assertAlmostEqual(sum(req.timings.values()), self.elapsed)

    async def test_title_fallback_preserves_existing_full_retry(self):
        self.fallback = "secondary"
        self.provider("primary", .3, .2, title_error=RuntimeError("synthetic title failure"))
        self.provider("secondary", .4, .1)
        req = self.request()
        result = await req.call(self.call_data())
        self.assertEqual(result, {"response_text": "scene secondary", "title": "title secondary"})
        self.assertAlmostEqual(req.timings["vision_request_s"], .7)
        self.assertAlmostEqual(req.timings["title_request_s"], .3)
        self.assertAlmostEqual(sum(req.timings.values()), self.elapsed)
        self.assertEqual(self.calls, [("primary", "vision"), ("primary", "title"), ("secondary", "vision"), ("secondary", "title")])

    async def test_cancellation_propagates(self):
        self.fallback = "secondary"
        self.provider("primary", .25, vision_error=asyncio.CancelledError())
        req = self.request()
        with self.assertRaises(asyncio.CancelledError):
            await req.call(self.call_data())
        self.assertAlmostEqual(req.timings["vision_request_s"], .25)
        self.assertEqual(req.timings["title_request_s"], 0)
        self.assertEqual(self.calls, [("primary", "vision")])

    async def test_request_isolation(self):
        a, b = self.request(), self.request()
        a.timings["vision_request_s"] = 8
        self.assertEqual(b.timings, {"vision_request_s": 0.0, "title_request_s": 0.0})

    async def test_failure_without_fallback(self):
        self.provider("primary", .2, vision_error=RuntimeError("synthetic failure"))
        req = self.request()
        with self.assertRaises(self.Error):
            await req.call(self.call_data())
        self.assertAlmostEqual(req.timings["vision_request_s"], .2)
        self.assertEqual(req.timings["title_request_s"], 0)

if __name__ == "__main__":
    unittest.main()
