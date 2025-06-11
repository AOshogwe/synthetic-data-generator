import re
import random
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Optional GIS libraries - install with: pip install geopy geopandas usaddress
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    import usaddress
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False
    logging.warning("GIS libraries not available. Install with: pip install geopy usaddress")


# Add this complete AddressSynthesizer class to your pipeline.py file

class AddressSynthesizer:
    """Handles address anonymization and synthesis with optional GIS integration"""

    def __init__(self, enable_gis=True):
        self.enable_gis = enable_gis and GIS_AVAILABLE
        self.geocoder = None
        self.street_cache = {}
        self.zip_cache = {}

        if self.enable_gis:
            try:
                self.geocoder = Nominatim(user_agent="synthetic_data_generator")
                logging.info("GIS integration enabled with Nominatim geocoder")
            except Exception as e:
                logging.warning(f"Could not initialize geocoder: {e}")
                self.enable_gis = False

        # Common street names for fallback synthesis
        self.common_street_names = [
            "Main St", "Oak Ave", "Maple Rd", "Park Blvd", "Cedar Ln", "Elm St",
            "Washington Ave", "Lincoln St", "Jefferson Rd", "Madison Ave", "Adams St",
            "Wilson Blvd", "Johnson St", "Williams Ave", "Brown Rd", "Davis St",
            "Miller Ave", "Moore St", "Taylor Rd", "Anderson Ave", "Thomas St",
            "Jackson Blvd", "White St", "Harris Ave", "Martin Rd", "Thompson St"
        ]

        # Sample zip codes by region (US-focused, can be expanded)
        self.sample_zip_codes = {
            'northeast': ['10001', '10002', '02101', '02102', '19101', '19102'],
            'southeast': ['30301', '30302', '33101', '33102', '28201', '28202'],
            'midwest': ['60601', '60602', '48201', '48202', '55401', '55402'],
            'southwest': ['75201', '75202', '85001', '85002', '87101', '87102'],
            'west': ['90210', '90211', '94101', '94102', '98101', '98102']
        }

    def identify_address_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that likely contain addresses"""
        address_columns = []

        for column in df.columns:
            column_lower = column.lower()

            # Check column name for address keywords
            if any(keyword in column_lower for keyword in
                   ['address', 'addr', 'street', 'location', 'residence']):
                address_columns.append(column)
                continue

            # Check data content for address patterns
            if self._looks_like_address_data(df[column]):
                address_columns.append(column)

        return address_columns

    def _looks_like_address_data(self, series: pd.Series) -> bool:
        """Check if a column contains address-like data"""
        # Sample a few values to check
        sample_values = series.dropna().head(10).astype(str)

        if len(sample_values) == 0:
            return False

        address_indicators = 0
        for value in sample_values:
            # Look for patterns common in addresses
            if any(pattern in value.lower() for pattern in
                   ['st ', 'ave ', 'rd ', 'blvd ', 'dr ', 'ln ', 'way ']):
                address_indicators += 1
            elif re.search(r'\d+.*\w+', value):  # Number followed by text
                address_indicators += 1
            elif re.search(r'\b\d{5}\b', value):  # 5-digit zip code
                address_indicators += 1

        # If more than 50% of samples look like addresses
        return address_indicators / len(sample_values) > 0.5

    def parse_address(self, address_str: str) -> Dict[str, str]:
        """Parse an address into components"""
        if not address_str or pd.isna(address_str):
            return {}

        address_str = str(address_str).strip()

        if self.enable_gis:
            try:
                # Use usaddress library for better parsing
                parsed = usaddress.tag(address_str)
                if parsed[1] == 'Street Address':
                    return parsed[0]
            except Exception as e:
                logging.debug(f"usaddress parsing failed for '{address_str}': {e}")

        # Fallback to regex-based parsing
        return self._regex_parse_address(address_str)

    def _regex_parse_address(self, address_str: str) -> Dict[str, str]:
        """Fallback address parsing using regex"""
        components = {}

        # Extract house number (digits at start)
        house_match = re.match(r'^(\d+)', address_str)
        if house_match:
            components['AddressNumber'] = house_match.group(1)

        # Extract zip code (5 digits, possibly with +4)
        zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', address_str)
        if zip_match:
            components['ZipCode'] = zip_match.group(1)

        # Extract state (2 letters before zip or at end)
        state_match = re.search(r'\b([A-Z]{2})\b(?:\s+\d{5})?', address_str)
        if state_match:
            components['StateName'] = state_match.group(1)

        # Street name is everything between house number and city/state
        street_pattern = r'^\d*\s*(.*?)(?:\s+[A-Z][a-z]+,?\s+[A-Z]{2}|\s+\d{5}|$)'
        street_match = re.search(street_pattern, address_str)
        if street_match:
            components['StreetName'] = street_match.group(1).strip()

        return components

    def anonymize_address(self, address_str: str, method: str = 'remove_house_number') -> str:
        """Anonymize an address using the specified method"""
        if not address_str or pd.isna(address_str):
            return address_str

        parsed = self.parse_address(address_str)

        if method == 'remove_house_number':
            return self._remove_house_number(address_str, parsed)
        elif method == 'street_only':
            return self._keep_street_only(parsed)
        elif method == 'city_state_only':
            return self._keep_city_state_only(parsed)
        elif method == 'zip_only':
            return self._keep_zip_only(parsed)
        elif method == 'general_area':
            return self._keep_general_area(parsed)
        elif method == 'synthesize_realistic':
            return self._synthesize_realistic_address(parsed)
        else:
            return address_str

    def _remove_house_number(self, original: str, parsed: Dict) -> str:
        """Remove just the house number, keep everything else"""
        if 'AddressNumber' in parsed:
            # Remove the house number from the beginning
            result = re.sub(r'^\d+\s*', '', original)
            return result.strip()
        return original

    def _keep_street_only(self, parsed: Dict) -> str:
        """Keep only the street name"""
        if 'StreetName' in parsed:
            return parsed['StreetName']
        elif 'StreetNamePreDirectional' in parsed and 'StreetNamePostType' in parsed:
            return f"{parsed.get('StreetNamePreDirectional', '')} {parsed.get('StreetNamePostType', '')}".strip()
        return "Street Name Unknown"

    def _keep_city_state_only(self, parsed: Dict) -> str:
        """Keep only city and state"""
        city = parsed.get('PlaceName', '')
        state = parsed.get('StateName', '')

        if city and state:
            return f"{city}, {state}"
        elif state:
            return state
        return "Location Unknown"

    def _keep_zip_only(self, parsed: Dict) -> str:
        """Keep only zip code"""
        return parsed.get('ZipCode', 'Unknown')

    def _keep_general_area(self, parsed: Dict) -> str:
        """Keep city, state, and zip for general area"""
        city = parsed.get('PlaceName', '')
        state = parsed.get('StateName', '')
        zip_code = parsed.get('ZipCode', '')

        parts = [part for part in [city, state, zip_code] if part]
        if len(parts) >= 2:
            return ', '.join(parts)
        elif parts:
            return parts[0]
        return "General Area Unknown"

    def _synthesize_realistic_address(self, parsed: Dict) -> str:
        """Generate a realistic synthetic address based on the original"""
        # Generate realistic house number
        house_num = random.randint(100, 9999)

        # Use a realistic street name
        if self.enable_gis and 'ZipCode' in parsed:
            street_name = self._get_realistic_street_for_zip(parsed['ZipCode'])
        else:
            street_name = random.choice(self.common_street_names)

        # Keep original city/state/zip if available, or generate realistic ones
        city = parsed.get('PlaceName', self._get_realistic_city())
        state = parsed.get('StateName', self._get_realistic_state())
        zip_code = parsed.get('ZipCode', self._get_realistic_zip())

        return f"{house_num} {street_name}, {city}, {state} {zip_code}"

    def _get_realistic_street_for_zip(self, zip_code: str) -> str:
        """Get a realistic street name for a given zip code using GIS"""
        if not self.enable_gis:
            return random.choice(self.common_street_names)

        # Cache results to avoid repeated API calls
        if zip_code in self.street_cache:
            return random.choice(self.street_cache[zip_code])

        try:
            # Geocode the zip code to get the area
            location = self.geocoder.geocode(zip_code, country_codes='us')
            if location:
                # For now, use common street names but this could be enhanced
                # with actual GIS queries to OpenStreetMap or other services
                self.street_cache[zip_code] = self.common_street_names[:10]
                return random.choice(self.street_cache[zip_code])
        except Exception as e:
            logging.debug(f"GIS lookup failed for zip {zip_code}: {e}")

        return random.choice(self.common_street_names)

    def _get_realistic_city(self) -> str:
        """Get a realistic city name"""
        cities = ['Springfield', 'Franklin', 'Georgetown', 'Madison', 'Washington',
                  'Arlington', 'Richmond', 'Oakland', 'Riverside', 'Fairview']
        return random.choice(cities)

    def _get_realistic_state(self) -> str:
        """Get a realistic state abbreviation"""
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        return random.choice(states)

    def _get_realistic_zip(self) -> str:
        """Get a realistic zip code"""
        all_zips = []
        for region_zips in self.sample_zip_codes.values():
            all_zips.extend(region_zips)
        return random.choice(all_zips)

    def process_address_column(self, series: pd.Series, method: str = 'remove_house_number') -> pd.Series:
        """Process an entire column of addresses"""
        logging.info(f"Processing address column with method: {method}")

        result = series.copy()
        processed_count = 0

        for idx, address in series.items():
            if pd.notna(address) and str(address).strip():
                try:
                    anonymized = self.anonymize_address(str(address), method)
                    result[idx] = anonymized
                    processed_count += 1
                except Exception as e:
                    logging.warning(f"Error processing address '{address}': {e}")
                    result[idx] = address  # Keep original if processing fails

        logging.info(f"Processed {processed_count} addresses using method '{method}'")
        return result