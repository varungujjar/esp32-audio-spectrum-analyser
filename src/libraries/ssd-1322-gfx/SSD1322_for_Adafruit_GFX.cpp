/*********************************************************************
SSD1322_for_Adafruit_GFX

This is a library for the Greyscale SSD1322 Controller based OLED
License: BSD

Forked in 09/2021 by venice1200 from the Adafruit SSD1327 Library 
and modfied for an SSD1322 OLED with 256x64 Pixel.
Other OLED Resolutions and Interfaces (I2C, 3SPI,6800,80xx) are currently (2021-09-14) not tested.


-----------------------------------------------------------
Original Adafruit Header for the SSD1327 OLED (BSD License)
-----------------------------------------------------------

This is a library for our Grayscale OLEDs based on SSD1327 drivers
  Pick one up today in the adafruit shop!
  ------> https://www.adafruit.com/products/4741

These displays use I2C or SPI to communicate

Adafruit invests time and resources providing this open source code,
please support Adafruit and open-source hardware by purchasing
products from Adafruit!

Original SSD1327 Library written by Limor Fried/Ladyada  for Adafruit Industries.
BSD license, check license.txt for more information
All text above, and the splash screen must be included in any redistribution

You will find the orginal Library here https://github.com/adafruit/Adafruit_SSD1327
*********************************************************************/


#include "SSD1322_for_Adafruit_GFX.h"
#include "splash.h"

// CONSTRUCTORS, DESTRUCTOR ------------------------------------------------

/*!
    @brief  Constructor for I2C-interfaced SSD1322 displays.
    @param  w
            Display width in pixels
    @param  h
            Display height in pixels
    @param  twi
            Pointer to an existing TwoWire instance (e.g. &Wire, the
            microcontroller's primary I2C bus).
    @param  rst_pin
            Reset pin (using Arduino pin numbering), or -1 if not used
            (some displays might be wired to share the microcontroller's
            reset pin).
    @param  clkDuring
            Speed (in Hz) for Wire transmissions in SSD1322 library calls.
            Defaults to 400000 (400 KHz), a known 'safe' value for most
            microcontrollers, and meets the SSD1322 datasheet spec.
            Some systems can operate I2C faster (800 KHz for ESP32, 1 MHz
            for many other 32-bit MCUs), and some (perhaps not all)
            SSD1322's can work with this -- so it's optionally be specified
            here and is not a default behavior. (Ignored if using pre-1.5.7
            Arduino software, which operates I2C at a fixed 100 KHz.)
    @param  clkAfter
            Speed (in Hz) for Wire transmissions following SSD1322 library
            calls. Defaults to 100000 (100 KHz), the default Arduino Wire
            speed. This is done rather than leaving it at the 'during' speed
            because other devices on the I2C bus might not be compatible
            with the faster rate. (Ignored if using pre-1.5.7 Arduino
            software, which operates I2C at a fixed 100 KHz.)
    @note   Call the object's begin() function before use -- buffer
            allocation is performed there!
*/
Adafruit_SSD1322::Adafruit_SSD1322(uint16_t w, uint16_t h, TwoWire *twi,
                                   int8_t rst_pin, uint32_t clkDuring,
                                   uint32_t clkAfter)
    : Adafruit_GrayOLED(4, w, h, twi, rst_pin, clkDuring, clkAfter) {}

/*!
    @brief  Constructor for SPI SSD1322 displays, using software (bitbang)
            SPI.
    @param  w
            Display width in pixels
    @param  h
            Display height in pixels
    @param  mosi_pin
            MOSI (master out, slave in) pin (using Arduino pin numbering).
            This transfers serial data from microcontroller to display.
    @param  sclk_pin
            SCLK (serial clock) pin (using Arduino pin numbering).
            This clocks each bit from MOSI.
    @param  dc_pin
            Data/command pin (using Arduino pin numbering), selects whether
            display is receiving commands (low) or data (high).
    @param  rst_pin
            Reset pin (using Arduino pin numbering), or -1 if not used
            (some displays might be wired to share the microcontroller's
            reset pin).
    @param  cs_pin
            Chip-select pin (using Arduino pin numbering) for sharing the
            bus with other devices. Active low.
    @note   Call the object's begin() function before use -- buffer
            allocation is performed there!
*/
Adafruit_SSD1322::Adafruit_SSD1322(uint16_t w, uint16_t h, int8_t mosi_pin,
                                   int8_t sclk_pin, int8_t dc_pin,
                                   int8_t rst_pin, int8_t cs_pin)
    : Adafruit_GrayOLED(4, w, h, mosi_pin, sclk_pin, dc_pin, rst_pin, cs_pin) {}

/*!
    @brief  Constructor for SPI SSD1322 displays, using native hardware SPI.
    @param  w
            Display width in pixels
    @param  h
            Display height in pixels
    @param  spi
            Pointer to an existing SPIClass instance (e.g. &SPI, the
            microcontroller's primary SPI bus).
    @param  dc_pin
            Data/command pin (using Arduino pin numbering), selects whether
            display is receiving commands (low) or data (high).
    @param  rst_pin
            Reset pin (using Arduino pin numbering), or -1 if not used
            (some displays might be wired to share the microcontroller's
            reset pin).
    @param  cs_pin
            Chip-select pin (using Arduino pin numbering) for sharing the
            bus with other devices. Active low.
    @param  bitrate
            SPI clock rate for transfers to this display. Default if
            unspecified is 8000000UL (8 MHz).
    @note   Call the object's begin() function before use -- buffer
            allocation is performed there!
*/
Adafruit_SSD1322::Adafruit_SSD1322(uint16_t w, uint16_t h, SPIClass *spi,
                                   int8_t dc_pin, int8_t rst_pin, int8_t cs_pin,
                                   uint32_t bitrate)
    : Adafruit_GrayOLED(4, w, h, spi, dc_pin, rst_pin, cs_pin, bitrate) {}

/*!
    @brief  Destructor for Adafruit_SSD1322 object.
*/
Adafruit_SSD1322::~Adafruit_SSD1322(void) {}

// ALLOCATE & INIT DISPLAY -------------------------------------------------

/*!
    @brief  Allocate RAM for image buffer, initialize peripherals and pins.
    @param  addr
            I2C address of corresponding SSD1322 display.
            SPI displays (hardware or software) do not use addresses, but
            this argument is still required (pass 0 or any value really,
            it will simply be ignored). Default if unspecified is 0.
    @param  reset
            If true, and if the reset pin passed to the constructor is
            valid, a hard reset will be performed before initializing the
            display. If using multiple SSD1322 displays on the same bus, and
            if they all share the same reset pin, you should only pass true
            on the first display being initialized, false on all others,
            else the already-initialized displays would be reset. Default if
            unspecified is true.
    @return true on successful allocation/init, false otherwise.
            Well-behaved code should check the return value before
            proceeding.
    @note   MUST call this function before any drawing or updates!
*/
bool Adafruit_SSD1322::begin(uint8_t addr, bool reset) {

  if (!Adafruit_GrayOLED::_init(addr, reset)) {
    return false;
  }


// Init Sequence
  oled_command(SSD1322_CMDLOCK);            // 0xFD
  oled_data(0x12);                          // 0x12 (Unlock OLED driver IC MCU interface)
  oled_command(SSD1322_DISPLAYOFF);         // 0xAE
  oled_command(SSD1322_DISPLAYCLK);         // 0xB3 (Set Oscillator Freq. & Display Clock Divider)
  oled_data(0x91);                          // 0x91 (Divide by 2, ~80 Frames/sec)
  oled_command(SSD1322_SETMULTIPLEX);       // 0xCA
  oled_data(0x3F);                          // 0x3F (1/64)
  oled_command(SSD1322_SETDISPLAYOFFSET);   // 0xA2
  oled_data(0x00);                          // 0x00 (No Offset)
  oled_command(SSD1322_SETSTARTLINE);       // 0xA1
  oled_data(0x00);                          // 0x00 (Start Line 0)
  oled_command(SSD1322_SEGREMAP);           // 0xA0
  oled_data(0x14);                          // 0x14 (Horizontal address increment, Disable Column Address Re-map, Enable Nibble Re-map)
  oled_data(0x11);                          // 0x11 (Enable Dual COM mode)
  oled_command(SSD1322_SETGPIO);            // 0xB5
  oled_data(0x00);                          // 0x00 (Input disabled)
  oled_command(SSD1322_FUNCSEL);            // 0xAB
  oled_data(0x01);                          // 0x01 (Enable internal VDD regulator)
  oled_command(SSD1322_DISPLAYENHA);        // 0xB4
  oled_data(0xA0);                          // 0xA0 (Enable External VSL)
  oled_data(0xFD);                          // 0xFD (Enhanced low GS display quality)
  oled_command(SSD1322_SETCONTRAST);        // 0xC1
  oled_data(0x05);                          // 0x05 (5)
  oled_command(SSD1322_MASTERCONTRAST);     // 0xC7
  oled_data(0x0F);                          // 0x0F (No change)
  oled_command(SSD1322_PHASELEN);           // 0xB1
  oled_data(0xE2);                          // 0xE2 (P1:5DCLKs,P2:14DCLKs)
  oled_command(SSD1322_DISPLAYENHB);        // 0xD1
  oled_data(0xA2);                          // 0xA2 (Normal)
  oled_data(0x20);                          // 0x20
  oled_command(SSD1322_PRECHARGE);          // 0xBB
  oled_data(0x1F);                          // 0x1F (0.6xVCC)
  oled_command(SSD1322_PRECHARGE2);         // 0xB6
  oled_data(0x08);                          // 0x08 (8 dclks)
  oled_command(SSD1322_SETVCOM);            // 0xBE
  oled_data(0x07);                          // 0x07 (0.86xVCC)
  oled_command(SSD1322_NORMALDISPLAY);      // 0xA6
  oled_command(SSD1322_EXITPARTDISPLAY);    // 0xA9
  oled_command(SSD1322_DISPLAYON);          // 0xAF

/*
      SSD1322_GRAYTABLE,
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
      0x07, 0x08, 0x10, 0x18, 0x20, 0x2f, 0x38, 0x3f,
*/

  delay(100);                               // 100ms delay recommended
  // oled_command(SSD1322_DISPLAYON);          // 0xAF
  // setContrast(0x2F);

  return true; // Success
}

/*!
    @brief  Do the actual writing of the internal frame buffer to display RAM
*/
void Adafruit_SSD1322::display(void) {
  // ESP8266 needs a periodic yield() call to avoid watchdog reset.
  // With the limited size of SSD1322 displays, and the fast bitrate
  // being used (1 MHz or more), I think one yield() immediately before
  // a screen write and one immediately after should cover it.  But if
  // not, if this becomes a problem, yields() might be added in the
  // 32-byte transfer condition below.
  yield();

  //uint16_t count = WIDTH * ((HEIGHT + 7) / 8);
  uint8_t *ptr = buffer;
  uint8_t dc_byte = 0x40;
  uint8_t rows = HEIGHT;

  uint8_t bytes_per_row = WIDTH / 2; // See fig 10-1 (64 bytes, 128 pixels)
  uint8_t maxbuff = 128;

  /*
  Serial.print("Window: (");
  Serial.print(window_x1);
  Serial.print(", ");
  Serial.print(window_y1);
  Serial.print(") -> (");
  Serial.print(window_x2);
  Serial.print(", ");
  Serial.print(window_y2);
  Serial.println(")");
  */

  int16_t row_start = min((int16_t)(bytes_per_row - 1), (int16_t)(window_x1 / 2));
  int16_t row_end = max((int16_t)0, (int16_t)(window_x2 / 2));

  int16_t first_row = min((int16_t)(rows - 1), (int16_t)window_y1);
  int16_t last_row = max((int16_t)0, (int16_t)window_y2);

  /*
  Serial.print("Rows: ");
  Serial.print(first_row);
  Serial.print(" -> ");
  Serial.println(last_row);

  Serial.print("Row start/end: ");
  Serial.print(row_start);
  Serial.print(" -> ");
  Serial.println(row_end);
  */

  if (i2c_dev) { // I2C
    // Set high speed clk
    i2c_dev->setSpeed(i2c_preclk);
    maxbuff = i2c_dev->maxBufferSize() - 1;
  }

  oled_command(SSD1322_SETROW);       // 0x75, Oled from Row 00h(0) to 3Fh(63)
  oled_data(0x00);                    // 00
  oled_data(0x3F);                    // 63, Testing, old value 127 (0x7F)
  oled_command(SSD1322_SETCOLUMN);    // 0x15, Oled from Column 1Ch(28) to 5B(91)
  oled_data(0x1C);                    // 28
  oled_data(0x5B);                    // 91
  oled_command(SSD1322_ENWRITEDATA);  // 0x5C

  for (uint8_t row = first_row; row <= last_row; row++) {
    uint8_t bytes_remaining = row_end - row_start + 1;
    ptr = buffer + (uint16_t)row * (uint16_t)bytes_per_row;
    // fast forward to dirty rectangle beginning
    ptr += row_start;

    while (bytes_remaining) {
      uint8_t to_write = min(bytes_remaining, maxbuff);
      if (i2c_dev) {
        i2c_dev->write(ptr, to_write, true, &dc_byte, 1);
      } else {
        digitalWrite(dcPin, HIGH);
        spi_dev->write(ptr, to_write);
      }
      ptr += to_write;
      bytes_remaining -= to_write;
      yield();
    }
  }

  if (i2c_dev) { // I2C
    // Set low speed clk
    i2c_dev->setSpeed(i2c_postclk);
  }

  // reset dirty window
  window_x1 = 0;
  window_y1 = 0;
  window_x2 = WIDTH - 1;
  window_y2 = HEIGHT - 1;
//  window_x1 = 1024;
//  window_y1 = 1024;
//  window_x2 = -1;
//  window_y2 = -1;

}

/*!
    @brief Issue single data byte to OLED, using I2C or hard/soft SPI as needed.
    @param c The single byte data
*/
void Adafruit_SSD1322::oled_data(uint8_t c) {
  if (i2c_dev) {                // I2C
    uint8_t buf[2] = {0x00, c}; // Co = 0, D/C = 0
    i2c_dev->write(buf, 2);
  } else { // SPI (hw or soft) -- transaction started in calling function
    digitalWrite(dcPin, HIGH);
    spi_dev->write(&c, 1);
  }
}

/*!
    @brief  Enable or disable display invert mode (white-on-black vs
            black-on-white). Handy for testing!
    @param  i
            If true, switch to invert mode (black-on-white), else normal
            mode (white-on-black).
*/
void Adafruit_SSD1322::invertDisplay(bool i) {
  oled_command(i ? SSD1322_INVERTDISPLAY : SSD1322_NORMALDISPLAY);
}

/*!
    @brief  Power Display off
*/
void Adafruit_SSD1322::displayOff(void) {
  oled_command(SSD1322_DISPLAYOFF);
}

/*!
    @brief  Power Display on
*/
void Adafruit_SSD1322::displayOn(void) {
  oled_command(SSD1322_DISPLAYON);
}

/*!
    @brief  Set all Pixel full off (GS=0)
*/
void Adafruit_SSD1322::allPixelOff(void) {
  oled_command(SSD1322_DISPLAYALLOFF);
}

/*!
    @brief  Set all Pixel full on (GS=15)
*/
void Adafruit_SSD1322::allPixelOn(void) {
  oled_command(SSD1322_DISPLAYALLON);
}

/*!
    @brief  Adjust the display contrast.
    @param  level The contrast level from 0 to 0xFF
    @note   This has an immediate effect on the display, no need to call the
            display() function -- buffer contents are not changed.
*/
void Adafruit_SSD1322::setContrast(uint8_t level) {
  oled_command(SSD1322_SETCONTRAST);
  oled_data(level);
}

/*!
    @brief  Returns the (Pointer to the) actual Font
    @param  None
*/
GFXfont * Adafruit_SSD1322::getFont(void) {
  return gfxFont;
}

/*!
    @brief  "Draw" a PROGMEM-resident Grayscale Bitmap Bytes 1:1 to the internal "buffer"
    @param  bitmap - The PROGMEM defined bitmap array
    @note   This has not an immediate effect on the display, you need to
            call the display() function
*/
void Adafruit_SSD1322::draw4bppBitmap(const uint8_t bitmap[]) {
  int16_t numOfBytes = WIDTH * HEIGHT / 2;
  uint8_t svalue1, svalue2, tvalue1, tvalue2;
  uint8_t *ptr1,*ptr2;
  int pheight, pwidth, i, j;

  switch (rotation) {
    case 0:  // 0°
      for (i=0; i<numOfBytes; i++) {
        ptr1=&buffer[i];                              // Set Pointer
        *ptr1=(uint8_t)pgm_read_byte(&bitmap[i]);     // Write Data
      }
    break;
    case 1:  // 90°
      // Example: Picture Bytes 0&32 => Buffer Position 127&255 + Nibble Changes N1/N2 + N3/N4 => N3/N1 + N4+N2
      pheight=WIDTH/2;           // calc rotated Height foor loop (256/2=128)
      pwidth=HEIGHT/2;           // calc rotated Width for loop   (64/2=32)
      for (j=0; j<pheight; j++) {
        for (i=0; i<pwidth; i++) {
          // Read Values
          svalue1=(uint8_t)pgm_read_byte(&bitmap[i+(j*2+0)*pwidth]);
          svalue2=(uint8_t)pgm_read_byte(&bitmap[i+(j*2+1)*pwidth]);
          // Calc Buffer Target Addresse 
          ptr1=&buffer[(i*2+1)*pheight-j-1];
          ptr2=&buffer[(i*2+2)*pheight-j-1];
          // Calc Target Values
          tvalue1=(0xF0 & svalue1) >> 4 | (0xF0 & svalue2);
          tvalue2=(0x0F & svalue1) | (0x0F & svalue2) << 4;
          // And write them
          *ptr1=tvalue1;
          *ptr2=tvalue2;
        }  // i
      }  // j
    break;
    case 2:  // 180°
      // Example: N1/N2 Picture Bytes 0...8191 => N2/N1 Buffer Bytes 8191...0
      for (i=0; i<numOfBytes; i++) {
        ptr1=&buffer[i];
        svalue1=(uint8_t)pgm_read_byte(&bitmap[numOfBytes-i-1]);
        tvalue1=(0xF0 & svalue1) >> 4 | (0x0F & svalue1) << 4;  // Swap High & Low Nibble
        *ptr1=tvalue1;
      }
    break;
    case 3:  // 270°
      // Example: Picture Bytes 0&32 => Buffer Position 7936&8064 + Nibble Changes N1/N2 + N3/N4 => N2/N4 + N1+N3
      pheight=WIDTH/2;           // calc rotated Height foor loop (256/2=128)
      pwidth=HEIGHT/2;           // calc rotated Width for loop   (64/2=32)
      for (j=0; j<pheight; j++) {
        for (i=0; i<pwidth; i++) {
          // Read Values
          svalue1=(uint8_t)pgm_read_byte(&bitmap[i+(j*2+0)*pwidth]);
          svalue2=(uint8_t)pgm_read_byte(&bitmap[i+(j*2+1)*pwidth]);
          // Calc Buffer Target Addresse 
          ptr1=&buffer[(HEIGHT-1)*pheight-(i*2+1)*pheight+j];
          ptr2=&buffer[(HEIGHT-1)*pheight-(i*2+0)*pheight+j];
          // Calc Target Values
          tvalue1=(0x0F & svalue1) << 4 | (0x0F & svalue2);
          tvalue2=(0xF0 & svalue1) | (0xF0 & svalue2) >> 4;
          // And write them
          *ptr1=tvalue1;
          *ptr2=tvalue2;
        }  // i
      }  // j
    break;
  } // endswitch
}

/*!
    @brief  "Draw" a RAM-resident Grayscale Bitmap Bytes 1:1 to the internal "buffer"
    @param  bitmap - The RAM defined bitmap array
    @note   This has not an immediate effect on the display, you need to
            call the display() function
*/
void Adafruit_SSD1322::draw4bppBitmap(uint8_t *bitmap) {
  int16_t numOfBytes = WIDTH * HEIGHT / 2;
  uint8_t svalue1, svalue2, tvalue1, tvalue2;
  uint8_t *ptr1, *ptr2;
  int pheight, pwidth, i, j;

  switch (rotation) {
    case 0:  // 0°
      for (i=0; i<numOfBytes; i++) {
        ptr1 = &buffer[i];
        *ptr1=(uint8_t)bitmap[i];
      }
    break;
    case 1:  // 90°
      // Example: Picture Bytes 0&32 => Buffer Position 127&255 + Nibble Changes N1/N2 + N3/N4 => N3/N1 + N4+N2
      pheight=WIDTH/2;           // calc rotated Height foor loop (256/2=128)
      pwidth=HEIGHT/2;           // calc rotated Width for loop   (64/2=32)
      for (j=0; j<pheight; j++) {
        for (i=0; i<pwidth; i++) {
          // Read Values
          svalue1=(uint8_t)bitmap[i+(j*2+0)*pwidth];
          svalue2=(uint8_t)bitmap[i+(j*2+1)*pwidth];
          // Calc Buffer Target Addresse 
          ptr1=&buffer[(i*2+1)*pheight-j-1];
          ptr2=&buffer[(i*2+2)*pheight-j-1];
          // Calc Target Values
          tvalue1=(0xF0 & svalue1) >> 4 | (0xF0 & svalue2);
          tvalue2=(0x0F & svalue1) | (0x0F & svalue2) << 4;
          // And write them
          *ptr1=tvalue1;
          *ptr2=tvalue2;
        }  // i
      }  // j
    break;
    case 2:  // 180°
      // Example: N1/N2 Picture Bytes 0...8191 => N2/N1 Buffer Bytes 8191...0
      for (i=0; i<numOfBytes; i++) {
        ptr1 = &buffer[i];
        svalue1=(uint8_t)bitmap[numOfBytes-i-1];
        tvalue1=(0xF0 & svalue1) >> 4 | (0x0F & svalue1) << 4;  // Swap High & Low Nibble
        *ptr1=tvalue1;
      }
    break;
    case 3:  // 270°
      // Example: Picture Bytes 0&32 => Buffer Position 7936&8064 + Nibble Changes N1/N2 + N3/N4 => N2/N4 + N1+N3
      pheight=WIDTH/2;           // calc rotated Height foor loop (256/2=128)
      pwidth=HEIGHT/2;           // calc rotated Width for loop   (64/2=32)
      for (j=0; j<pheight; j++) {
        for (i=0; i<pwidth; i++) {
          // Read Values
          svalue1=(uint8_t)bitmap[i+(j*2+0)*pwidth];
          svalue2=(uint8_t)bitmap[i+(j*2+1)*pwidth];
          // Calc Buffer Target Addresse 
          ptr1=&buffer[(HEIGHT-1)*pheight-(i*2+1)*pheight+j];
          ptr2=&buffer[(HEIGHT-1)*pheight-(i*2+0)*pheight+j];
          // Calc Target Values
          tvalue1=(0x0F & svalue1) << 4 | (0x0F & svalue2);
          tvalue2=(0xF0 & svalue1) | (0xF0 & svalue2) >> 4;
          // And write them
          *ptr1=tvalue1;
          *ptr2=tvalue2;
        }  // i
      }  // j
    break;
  } // endswitch
}
