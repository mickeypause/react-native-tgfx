// react-native-tgfx example — shapes, strokes, gradients, image and text through <Canvas>.
import { memo, useEffect, useState } from 'react';
import {
  Pressable,
  Text as RNText,
  ScrollView,
  StyleSheet,
  View,
} from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import {
  Canvas,
  Circle,
  Group,
  Image,
  LinearGradient,
  Line,
  Oval,
  Path,
  Paint,
  RadialGradient,
  Rect,
  RoundedRect,
  Text,
  useImage,
  useNode,
} from 'react-native-tgfx';
import { FilterSection } from './src/filterExamples';
import { ImageSection } from './src/imageExamples';
import { PathSection } from './src/pathExamples';
import { LogoSection } from './src/logoExample';
import { LogoScreen } from './src/logoScreen';
import { GroupSection } from './src/groupExamples';
import { PathEffectSection } from './src/pathEffectExamples';
import { TextSection } from './src/textExamples';
import { VideoSection } from './src/videoExamples';
import {
  BarChart,
  CircleField,
  Heart,
  RegularPolygon,
  Spiral,
  Star,
  Wave,
} from './src/customShapes';
import { AnimatedSection } from './src/animatedExamples';
import { PanScreen } from './src/panExamples';
import { ImperativeSection } from './src/imperativeExample';
import { ImperativeSlotsSection } from './src/imperativeSlotsExample';
import { SnapshotSection } from './src/snapshotExample';
import {
  BlurPanel,
  CheckerReference,
  CyberpunkPanel,
  ChildShaderPanels,
  HelperCollisionPanels,
  SdCirclePanel,
} from './src/shaderExamples';
import { TgfxCameraEffects } from './src/TgfxCameraEffects';
import { LittlePrinceScreen } from './src/littlePrince';
import { ScrollableShapesScreen } from './src/scrollableShapes';
import { PhysicsScreen } from './src/physics';
import { CubeScreen } from './src/cubeScreen';

// Memoised: the tab switch re-renders Screens, and re-rendering the hidden
// gallery's dozens of canvases is pure waste on the camera-start path.
const Demo = memo(function Demo() {
  const insets = useSafeAreaInsets();
  const img = useImage(require('./src/assets/sample.png'));

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={{ paddingTop: insets.top + 8, paddingBottom: 32 }}
    >
      <RNText style={styles.title}>react-native-tgfx</RNText>
      <RNText style={styles.subtitle}>declarative tgfx canvas · Nitro</RNText>

      <Section label="Shapes (fill)">
        <Rect x={0} y={0} width={80} height={56} color="#4F8EF7" />
        <RoundedRect
          x={96}
          y={0}
          width={80}
          height={56}
          r={14}
          color="#34C759"
        />
        <Circle cx={216} cy={28} r={28} color="#FF9500" />
        <Oval x={256} y={0} width={92} height={56} color="#AF52DE" />
      </Section>

      <Section label="Strokes (cap / join / path)">
        <Circle
          cx={28}
          cy={28}
          r={24}
          color="#FF3B30"
          style="stroke"
          strokeWidth={6}
        />
        <RoundedRect
          x={72}
          y={2}
          width={84}
          height={52}
          r={12}
          color="#111"
          style="stroke"
          strokeWidth={4}
        />
        <Line
          x1={176}
          y1={6}
          x2={250}
          y2={52}
          color="#1C1C1E"
          style="stroke"
          strokeWidth={8}
          strokeCap="round"
        />
        <Path
          path="M262 50 q24 -52 48 0 t48 0"
          color="#5856D6"
          style="stroke"
          strokeWidth={5}
          strokeJoin="round"
        />
      </Section>

      <Section label="Miter limit · same spike, cut back at 1, kept at 20">
        {[1, 20].map((limit, i) => (
          <Path
            key={limit}
            path={`M${i * 140 + 30} 56 L${i * 140 + 44} 8 L${i * 140 + 58} 56`}
            color="#FF3B30"
            style="stroke"
            strokeWidth={10}
            strokeJoin="miter"
            strokeMiter={limit}
          />
        ))}
      </Section>

      <Section label="color(display-p3) — P3 pairs above their sRGB equivalents">
        {[
          ['color(display-p3 1 0 0)', '#ff0000'],
          ['color(display-p3 0 1 0)', '#00ff00'],
          ['color(display-p3 0 0 1)', '#0000ff'],
          ['color(display-p3 1 0.8 0)', '#ffcc00'],
        ].map(([p3, srgb], i) => (
          <Group key={p3} transform={[{ translateX: i * 88 }]}>
            <Rect x={0} y={0} width={80} height={32} color={p3} />
            <Rect x={0} y={36} width={80} height={32} color={srgb} />
          </Group>
        ))}
      </Section>

      <Section label="Gradients">
        <RoundedRect x={0} y={0} width={170} height={64} r={16}>
          <LinearGradient
            start={{ x: 0, y: 0 }}
            end={{ x: 170, y: 64 }}
            colors={['#FF6CAB', '#7366FF']}
          />
        </RoundedRect>
        <Circle cx={250} cy={32} r={32}>
          <RadialGradient
            center={{ x: 250, y: 32 }}
            r={32}
            colors={['#FFE259', '#FFA751']}
          />
        </Circle>
        <Group transform={[{ translateX: 300 }]}>
          <Rect x={0} y={0} width={56} height={64}>
            <Paint>
              <LinearGradient
                start={{ x: 0, y: 0 }}
                end={{ x: 0, y: 64 }}
                colors={['#11998e', '#38ef7d']}
              />
            </Paint>
          </Rect>
        </Group>
      </Section>

      <Section label="Image (fit: cover / contain)">
        {img != null && (
          <Image x={0} y={0} width={88} height={88} image={img} fit="cover" />
        )}
        {img != null && (
          <Image
            x={104}
            y={0}
            width={88}
            height={88}
            image={img}
            fit="contain"
          />
        )}
        {img != null && (
          <Group transform={[{ translateX: 208 }, { rotate: '8deg' }]}>
            <Image x={0} y={0} width={88} height={88} image={img} fit="cover" />
          </Group>
        )}
      </Section>

      <Section label="Text">
        <Text
          x={0}
          y={26}
          text="Shapes · Strokes · Image · Text"
          fontSize={20}
          color="#111"
        />
        <Text
          x={0}
          y={56}
          text="rendered through tgfx + Nitro"
          fontSize={16}
          color="#888"
        />
        <Group transform={[{ translateX: 0 }, { translateY: 72 }]}>
          <Text
            x={0}
            y={20}
            text="GPU 2D · Metal / OpenGL"
            fontSize={15}
            color="#4F8EF7"
          />
        </Group>
      </Section>

      <TextSection />

      <PathSection />

      <LogoSection />

      <GroupSection />

      <PathEffectSection />

      <Section label="Custom shapes (composed from primitives)" height={150}>
        <Star cx={42} cy={74} outerR={36} points={5} color="#FF9500" />
        <Star
          cx={124}
          cy={74}
          outerR={34}
          innerR={22}
          points={6}
          color="#FF3B30"
          style="stroke"
          strokeWidth={3}
          strokeJoin="round"
        />
        <RegularPolygon cx={206} cy={74} r={36} sides={6} color="#34C759" />
        <RegularPolygon
          cx={282}
          cy={74}
          r={36}
          sides={3}
          color="#5856D6"
          style="stroke"
          strokeWidth={3}
          strokeJoin="round"
        />
        <Heart cx={344} cy={70} size={42} color="#FF2D55" />
      </Section>

      <Section label="Procedural / generative (hsl palettes)" height={220}>
        <Spiral cx={80} cy={102} count={120} spacing={7} dotR={3.5} />
        <CircleField x={175} y={16} cols={5} rows={6} cell={28} />
        <Wave
          x={8}
          y={200}
          width={345}
          amplitude={12}
          wavelength={80}
          color="#0A84FF"
          strokeWidth={3}
        />
      </Section>

      <Section label="Bar chart from data" height={140}>
        <BarChart
          x={8}
          y={10}
          width={350}
          height={118}
          data={[12, 28, 18, 40, 33, 52, 22, 36, 47, 30]}
        />
      </Section>

      <Section label="Runtime shader · blur kernel (procedural)" height={140}>
        <CheckerReference x={0} y={0} width={168} height={120} />
        <BlurPanel x={182} y={0} width={168} height={120} radius={3} />
      </Section>

      <FilterSection />
      <ImageSection />
      <VideoSection />
      <Section label="Runtime shader · helper function (sdCircle)" height={140}>
        <SdCirclePanel x={0} y={0} width={168} height={120} />
        <HelperCollisionPanels x={182} y={0} width={168} height={120} />
      </Section>

      <Section
        label="Runtime shader · child shaders (eval / sample)"
        height={140}
      >
        <ChildShaderPanels x={0} y={0} width={350} height={120} />
      </Section>

      <ShaderScreenSection />

      <View style={styles.section}>
        <RNText style={styles.sectionLabel}>
          Animated · Reanimated UI-thread driver (zero JS / frame)
        </RNText>
        <AnimatedSection />
      </View>

      <ImperativeSection />

      <ImperativeSlotsSection />

      <SnapshotSection />
    </ScrollView>
  );
});

function ShaderScreenSection() {
  // Patched through the node handle: 20 ticks/s as setState was a full React
  // commit each — the single largest JS cost in the whole app's Perfetto trace.
  const shader = useNode();

  useEffect(() => {
    const started = Date.now();
    const id = setInterval(
      () => shader.set({ uniforms: { uTime: (Date.now() - started) / 1000 } }),
      50,
    );
    return () => clearInterval(id);
  }, [shader]);

  return (
    <View style={styles.section}>
      <RNText style={styles.sectionLabel}>
        Runtime shader · cyberpunk screen (uniform patched per tick, no
        re-render)
      </RNText>
      <Canvas style={[styles.canvas, { height: 200 }]}>
        <CyberpunkPanel
          x={0}
          y={0}
          width={350}
          height={180}
          shaderRef={shader}
        />
      </Canvas>
    </View>
  );
}

function Section({
  label,
  children,
  height = 100,
}: {
  label: string;
  children: React.ReactNode;
  height?: number;
}) {
  return (
    <View style={styles.section}>
      <RNText style={styles.sectionLabel}>{label}</RNText>
      <Canvas composite="texture" style={[styles.canvas, { height }]}>
        {children}
      </Canvas>
    </View>
  );
}

const SCREENS = [
  'gallery',
  'logo',
  'pan',
  'effects',
  'prince',
  'shapes',
  'physics',
  'cube',
] as const;
type Screen = (typeof SCREENS)[number];

function Screens() {
  const [screen, setScreen] = useState<Screen>('gallery');
  // Trails one frame behind: unmounting the gallery's many canvases is slow, so hide first, unmount hidden.
  const [mounted, setMounted] = useState<Screen>('gallery');
  const insets = useSafeAreaInsets();

  useEffect(() => {
    if (mounted !== screen) {
      const id = requestAnimationFrame(() => setMounted(screen));
      return () => cancelAnimationFrame(id);
    }
  }, [screen, mounted]);

  return (
    <View style={styles.root}>
      {/* One container per screen, each with a fixed testID: a testID that
          changes on a mounted view does not reach the native view. */}
      {screen === 'gallery' || mounted === 'gallery' ? (
        <View
          style={[styles.screen, screen !== 'gallery' && styles.hiddenScreen]}
          testID="screen-gallery"
        >
          <Demo />
        </View>
      ) : null}
      {screen === 'logo' ? (
        <View style={styles.screenDark} testID="screen-logo">
          <LogoScreen />
        </View>
      ) : null}
      {screen === 'pan' ? (
        <View style={styles.screen} testID="screen-pan">
          <PanScreen />
        </View>
      ) : null}
      {screen === 'effects' ? (
        <View style={styles.screenDark} testID="screen-effects">
          <TgfxCameraEffects />
        </View>
      ) : null}
      {screen === 'prince' ? (
        <View style={styles.screen} testID="screen-prince">
          <LittlePrinceScreen />
        </View>
      ) : null}
      {screen === 'shapes' ? (
        <View style={styles.screenDark} testID="screen-shapes">
          <ScrollableShapesScreen />
        </View>
      ) : null}
      {screen === 'physics' ? (
        <View style={styles.screen} testID="screen-physics">
          <PhysicsScreen />
        </View>
      ) : null}
      {screen === 'cube' ? (
        <View style={styles.screenDark} testID="screen-cube">
          <CubeScreen />
        </View>
      ) : null}
      <View style={[styles.tabs, { paddingBottom: insets.bottom + 8 }]}>
        {SCREENS.map(name => (
          <Pressable
            key={name}
            testID={`tab-${name}`}
            accessible
            accessibilityRole="button"
            accessibilityLabel={name}
            accessibilityState={{ selected: screen === name }}
            style={[styles.tab, screen === name && styles.tabOn]}
            onPress={() => setScreen(name)}
          >
            <RNText
              style={[styles.tabText, screen === name && styles.tabTextOn]}
            >
              {name}
            </RNText>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

export default function App() {
  return (
    <GestureHandlerRootView style={styles.root}>
      <SafeAreaProvider>
        <Screens />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  screen: { flex: 1 },
  screenDark: { flex: 1, backgroundColor: '#000' },
  hiddenScreen: { display: 'none' },
  tabs: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
    paddingTop: 8,
    backgroundColor: '#111',
  },
  tab: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#fff4',
  },
  tabOn: { backgroundColor: '#fff2', borderColor: '#fff' },
  tabText: { color: '#fff9', fontSize: 14, fontWeight: '600' },
  tabTextOn: { color: '#fff' },
  scroll: { flex: 1, backgroundColor: '#F2F2F7' },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#111',
    paddingHorizontal: 16,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  section: { paddingHorizontal: 16, marginTop: 14 },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
    marginBottom: 6,
  },
  canvas: {
    height: 100,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
  },
});
