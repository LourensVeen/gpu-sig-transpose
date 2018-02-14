#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <x86intrin.h>

#define NPOLS 4
#define NCHANS 4
#define NSAMPS 500

char *transposed;
char *page;

/* Lower bound on timings by a single memcpy
 * gives ~1.19 seconds for 10 iterations, 1757 MB  /0.11s ~= 1600 MB/s
   [root@laptopjisk ~]# dmidecode -t 17
   # dmidecode 3.1
   Getting SMBIOS data from sysfs.
   SMBIOS 3.0.0 present.

   Handle 0x0039, DMI type 17, 40 bytes
   Memory Device
               Array Handle: 0x0038
               Error Information Handle: Not Provided
               Total Width: 64 bits
               Data Width: 64 bits
               Size: 8192 MB
               Form Factor: Row Of Chips
               Set: None
               Locator: System Board Memory
               Bank Locator: BANK 0
               Type: LPDDR3
               Type Detail: Synchronous Unbuffered (Unregistered)
               Speed: 1867 MT/s
               Manufacturer: Micron
               Serial Number: 00000000
               Asset Tag: 9876543210
               Part Number: MT52L1G32D4PG-107 
               Rank: 2
               Configured Clock Speed: 1867 MT/s
               Minimum Voltage: 1.25 V
               Maximum Voltage: 1.25 V
               Configured Voltage: 1.2 V
*/


void deinterleave_block(const unsigned char * block, unsigned char * transposed) {
    const __m128i r0 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r1 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r2 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r3 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r4 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r5 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r6 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r7 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r8 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r9 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r10 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r11 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r12 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r13 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r14 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r15 = _mm_loadu_si128((__m128i*)block); block += 16;
    const __m128i r16 = _mm_loadu_si128((__m128i*)block); block += 16;

    const __m128i s0 = _mm_unpacklo_epi8(r0, r1);
    const __m128i s1 = _mm_unpackhi_epi8(r0, r1);
    const __m128i s2 = _mm_unpacklo_epi8(r2, r3);
    const __m128i s3 = _mm_unpackhi_epi8(r2, r3);
    const __m128i s4 = _mm_unpacklo_epi8(r4, r5);
    const __m128i s5 = _mm_unpackhi_epi8(r4, r5);
    const __m128i s6 = _mm_unpacklo_epi8(r6, r7);
    const __m128i s7 = _mm_unpackhi_epi8(r6, r7);
    const __m128i s8 = _mm_unpacklo_epi8(r8, r9);
    const __m128i s9 = _mm_unpackhi_epi8(r8, r9);
    const __m128i s10 = _mm_unpacklo_epi8(r10, r11);
    const __m128i s11 = _mm_unpackhi_epi8(r10, r11);
    const __m128i s12 = _mm_unpacklo_epi8(r12, r13);
    const __m128i s13 = _mm_unpackhi_epi8(r12, r13);
    const __m128i s14 = _mm_unpacklo_epi8(r14, r15);
    const __m128i s15 = _mm_unpackhi_epi8(r14, r15);

    const __m128i t0 = _mm_unpacklo_epi16(s0, s2);
    const __m128i t1 = _mm_unpackhi_epi16(s0, s2);
    const __m128i t2 = _mm_unpacklo_epi16(s1, s3);
    const __m128i t3 = _mm_unpackhi_epi16(s1, s3);
    const __m128i t4 = _mm_unpacklo_epi16(s4, s6);
    const __m128i t5 = _mm_unpackhi_epi16(s4, s6);
    const __m128i t6 = _mm_unpacklo_epi16(s5, s7);
    const __m128i t7 = _mm_unpackhi_epi16(s5, s7);
    const __m128i t8 = _mm_unpacklo_epi16(s8, s10);
    const __m128i t9 = _mm_unpackhi_epi16(s8, s10);
    const __m128i t10 = _mm_unpacklo_epi16(s9, s11);
    const __m128i t11 = _mm_unpackhi_epi16(s9, s11);
    const __m128i t12 = _mm_unpacklo_epi16(s12, s14);
    const __m128i t13 = _mm_unpackhi_epi16(s12, s14);
    const __m128i t14 = _mm_unpacklo_epi16(s13, s15);
    const __m128i t15 = _mm_unpackhi_epi16(s13, s15);

    const __m128i u0 = _mm_unpacklo_epi32(t0, t4);
    const __m128i u1 = _mm_unpackhi_epi32(t0, t4);
    const __m128i u2 = _mm_unpacklo_epi32(t1, t5);
    const __m128i u3 = _mm_unpackhi_epi32(t1, t5);
    const __m128i u4 = _mm_unpacklo_epi32(t2, t6);
    const __m128i u5 = _mm_unpackhi_epi32(t2, t6);
    const __m128i u6 = _mm_unpacklo_epi32(t3, t7);
    const __m128i u7 = _mm_unpackhi_epi32(t3, t7);
    const __m128i u8 = _mm_unpacklo_epi32(t8, t12);
    const __m128i u9 = _mm_unpackhi_epi32(t8, t12);
    const __m128i u10 = _mm_unpacklo_epi32(t9, t13);
    const __m128i u11 = _mm_unpackhi_epi32(t9, t13);
    const __m128i u12 = _mm_unpacklo_epi32(t10, t14);
    const __m128i u13 = _mm_unpackhi_epi32(t10, t14);
    const __m128i u14 = _mm_unpacklo_epi32(t11, t15);
    const __m128i u15 = _mm_unpackhi_epi32(t11, t15);

    const __m128i v0 = _mm_unpacklo_epi64(u0, u8);
    const __m128i v1 = _mm_unpackhi_epi64(u0, u8);
    const __m128i v2 = _mm_unpacklo_epi64(u1, u9);
    const __m128i v3 = _mm_unpackhi_epi64(u1, u9);
    const __m128i v4 = _mm_unpacklo_epi64(u2, u10);
    const __m128i v5 = _mm_unpackhi_epi64(u2, u10);
    const __m128i v6 = _mm_unpacklo_epi64(u3, u11);
    const __m128i v7 = _mm_unpackhi_epi64(u3, u11);
    const __m128i v8 = _mm_unpacklo_epi64(u4, u12);
    const __m128i v9 = _mm_unpackhi_epi64(u4, u12);
    const __m128i v10 = _mm_unpacklo_epi64(u5, u13);
    const __m128i v11 = _mm_unpackhi_epi64(u5, u13);
    const __m128i v12 = _mm_unpacklo_epi64(u6, u14);
    const __m128i v13 = _mm_unpackhi_epi64(u6, u14);
    const __m128i v14 = _mm_unpacklo_epi64(u7, u15);
    const __m128i v15 = _mm_unpackhi_epi64(u7, u15);

    *((__m128i*)transposed) = v0; transposed += 16;
    *((__m128i*)transposed) = v1; transposed += 16;
    *((__m128i*)transposed) = v2; transposed += 16;
    *((__m128i*)transposed) = v3; transposed += 16;
    *((__m128i*)transposed) = v4; transposed += 16;
    *((__m128i*)transposed) = v5; transposed += 16;
    *((__m128i*)transposed) = v6; transposed += 16;
    *((__m128i*)transposed) = v7; transposed += 16;
    *((__m128i*)transposed) = v8; transposed += 16;
    *((__m128i*)transposed) = v9; transposed += 16;
    *((__m128i*)transposed) = v10; transposed += 16;
    *((__m128i*)transposed) = v11; transposed += 16;
    *((__m128i*)transposed) = v12; transposed += 16;
    *((__m128i*)transposed) = v13; transposed += 16;
    *((__m128i*)transposed) = v14; transposed += 16;
    *((__m128i*)transposed) = v15; transposed += 16;
}



/**
 * Deinterleave (transpose) an IQUV ring buffer page to the ordering needed for FITS files
 * Note that this is probably a slow function, and is not meant to be run real-time
 *
 *  data in:   tab, channel/4, time/500 packets of time,channel,pn
 *  data  out: tab, channel, pol, time
 *
 * Suggested use is:
 *   1. realtime: ringbuffer -> [trigger] -> dada_dbdisk
 *   2. offline: dada_dbdisk -> ringbuffer -> dadafits
 *
 *  @param {const unsigned char *} page    Ringbuffer page with interleaved data
 *  @param {int} ntabs                     Number of tabs
 *  @param {int} nchannels                 Number of channels
 *  @param {int} npackets                  Number of packets per sequence
 */
void deinterleave (const char *page, const int ntabs, const int nchannels, const int npackets) {
  const char *block = page;
  char *trans = transposed;

  int total_blocks = (ntabs * nchannels * NSAMPS * npackets * NPOLS) / 256;
  int blocknum;
  for (blocknum = 0; blocknum < total_blocks; blocknum++) {
    deinterleave_block(block, trans);
    // memcpy(trans, block, 256);
    block += 256;
    trans += 256;
  }
  // printf("%ld\n", (trans - transposed));
}


int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "Need 3 arguments: ntabs, nchannels, npackets\n");
    exit(EXIT_FAILURE);
  }

  int ntabs = atoi(argv[1]);
  int nchannels = atoi(argv[2]);
  int npackets = atoi(argv[3]);

  size_t mysize = ntabs * nchannels * NSAMPS * npackets * NPOLS;
  printf("% 4i % 4i % 4i %6.2fMB\n", ntabs, nchannels, npackets, mysize / (1024.0*1024.0));

  transposed = malloc(mysize);
  page = malloc(mysize);

  // fill with test data
  /*
  unsigned char * p = page;
  int j, k, l;
  int total_blocks = (ntabs * nchannels * NSAMPS * npackets * NPOLS) / 256;
  for (l = 0; l < total_blocks; ++l)
    for (k = 0; k < 16; ++ k)
      for (j = 0; j < 16; ++j)
        *p++ = k * 16 + j;
  */

  // transpose
  int i;
  for (i=0; i<10; i++) {
    deinterleave(page, ntabs, nchannels, npackets);
    //memcpy(transposed, page, mysize);
  }

  // check if it's correct
  /*
  unsigned char * t = transposed;
  for (l = 0; l < 1; ++l)
    for (k = 0; k < 16; ++ k)
      for (j = 0; j < 16; ++j) {
        if (*t != j * 16 + k) {
          printf("Error at %d %d %d: %d\n", l, k, j, *t);
          exit(1);
        }
        ++t;
      }
  */

  free(page);
  free(transposed);

  exit(EXIT_SUCCESS);
}
