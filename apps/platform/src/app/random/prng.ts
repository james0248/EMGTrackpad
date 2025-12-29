/**
 * Seeded pseudo-random number generator (Mulberry32)
 * Simple, fast, and produces decent quality randomness for our use case.
 */
export class PRNG {
  private state: number;

  constructor(seed: number) {
    this.state = seed;
  }

  /**
   * Returns a random float in [0, 1)
   */
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * Returns a random integer in [min, max] inclusive
   */
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  /**
   * Returns a random float in [min, max)
   */
  nextFloat(min: number, max: number): number {
    return this.next() * (max - min) + min;
  }

  /**
   * Returns a random boolean with given probability of true
   */
  nextBool(probability = 0.5): boolean {
    return this.next() < probability;
  }

  /**
   * Picks a random element from an array
   */
  pick<T>(arr: T[]): T {
    return arr[this.nextInt(0, arr.length - 1)]!;
  }

  /**
   * Shuffles an array in place (Fisher-Yates)
   */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextInt(0, i);
      [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }
    return arr;
  }
}

/**
 * Generate a random seed based on current time
 */
export function generateSeed(): number {
  return Math.floor(Math.random() * 2147483647);
}
