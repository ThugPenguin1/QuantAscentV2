import requests
import time
import hmac
import hashlib


class RoostooAPI:
    def __init__(self, api_key, api_secret, base_url="https://mock-api.roostoo.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.last_call_time = 0
        self.min_interval = 2.1  # 30 calls/minute protection

    def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

    def _timestamp(self):
        return str(int(time.time() * 1000))

    def _sign(self, payload):
        payload = dict(payload)
        payload['timestamp'] = self._timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        return headers, payload, total_params

    def server_time(self):
        try:
            r = requests.get(f"{self.base_url}/v3/serverTime", timeout=10)
            return r.json()
        except Exception as e:
            print(f"[API ERROR] server_time: {e}")
            return None

    def exchange_info(self):
        try:
            r = requests.get(f"{self.base_url}/v3/exchangeInfo", timeout=10)
            return r.json()
        except Exception as e:
            print(f"[API ERROR] exchange_info: {e}")
            return None

    def ticker(self, pair=None):
        self._rate_limit()
        params = {"timestamp": self._timestamp()}
        if pair:
            params["pair"] = pair
        try:
            r = requests.get(f"{self.base_url}/v3/ticker", params=params, timeout=10)
            data = r.json()
            if data.get("Success"):
                return data.get("Data", {})
            print(f"[API] ticker error: {data.get('ErrMsg')}")
            return None
        except Exception as e:
            print(f"[API ERROR] ticker: {e}")
            return None

    def balance(self):
        self._rate_limit()
        headers, payload, _ = self._sign({})
        try:
            r = requests.get(
                f"{self.base_url}/v3/balance",
                headers=headers,
                params=payload,
                timeout=10
            )

            text = r.text.strip()
            if not text:
                print("[API ERROR] balance: empty response")
                return None

            try:
                data = r.json()
            except Exception:
                print(f"[API ERROR] balance non-JSON response: {text[:500]}")
                return None

            if data.get("Success"):
                return data.get("SpotWallet") or data.get("Wallet", {})

            print(f"[API] balance error: {data.get('ErrMsg')}")
            return None

        except Exception as e:
            print(f"[API ERROR] balance: {e}")
            return None
    def place_order(self, pair, side, quantity, price=None, order_type=None):
        self._rate_limit()

        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"

        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }

        if order_type.upper() == "LIMIT" and price is not None:
            payload["price"] = str(price)

        headers, _, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            r = requests.post(
                f"{self.base_url}/v3/place_order",
                headers=headers,
                data=total_params,
                timeout=15
            )
            data = r.json()

            if data.get("Success"):
                detail = data.get("OrderDetail", {})
                print(
                    f"  [ORDER] {side.upper()} {quantity} {pair} @ "
                    f"{'$'+str(price) if price else 'MARKET'} -> "
                    f"Status: {detail.get('Status')} "
                    f"ID: {detail.get('OrderID')}"
                )
                return data

            print(f"  [ORDER FAIL] {pair} {side}: {data.get('ErrMsg')}")
            return data

        except Exception as e:
            print(f"[API ERROR] place_order: {e}")
            return None

    def cancel_order(self, order_id=None, pair=None):
        self._rate_limit()
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair

        headers, _, total_params = self._sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            r = requests.post(
                f"{self.base_url}/v3/cancel_order",
                headers=headers,
                data=total_params,
                timeout=10
            )
            return r.json()
        except Exception as e:
            print(f"[API ERROR] cancel_order: {e}")
            return None

    def pending_count(self):
        self._rate_limit()
        headers, payload, _ = self._sign({})
        try:
            r = requests.get(
                f"{self.base_url}/v3/pending_count",
                headers=headers,
                params=payload,
                timeout=10
            )
            return r.json()
        except Exception as e:
            print(f"[API ERROR] pending_count: {e}")
            return None