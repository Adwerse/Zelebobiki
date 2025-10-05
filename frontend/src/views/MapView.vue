<script setup>
import { Deck, TileLayer, BitmapLayer, MapView, HeatmapLayer, HexagonLayer } from 'deck.gl'
import { onBeforeUnmount, onMounted, onUnmounted, ref, watch } from 'vue'

const deckContainer = ref(null);
const coords = ref(null);
const deck = ref(null);
const popUp = ref(null);

const INITIAL_VIEW_STATE = {
  longitude: 0,
  latitude: 0,
  zoom: 2,
  pitch: 0,
  bearing: 0
};

onMounted(() => {
  document.body.classList.add('no-scroll')
})
onUnmounted(() => {
  document.body.classList.remove('no-scroll')
})

const oceanBasemap = new TileLayer({
  data: 'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
  minZoom: 0,
  maxZoom: 5,
  tileSize: 256,

  renderSubLayers: props => {
    const {
      bbox: {west, south, east, north}
    } = props.tile;
    return new BitmapLayer(props, {
      data: null,
      image: props.data,
      bounds: [west, south, east, north]
    });
  }
});

const COLOR_RANGE = [
  [255, 0, 0],
  [255, 0, 0],
  [255, 0, 0],
  [255, 0, 0],
  [255, 0, 0],
  [255, 0, 0],
];

// const layer = new HeatmapLayer({
//   id: 'HeatmapLayer',
//   data: 'sharks.json',
//   aggregation: 'SUM',
//   getPosition: d => [d.lon, d.lat],
//   radiusPixels: 25,
//   intensity: 2.0,
//   threshold: 0.02,                    // опускаем порог, чтобы одиночные точки проявлялись
//   colorRange: [
//     [0,   0,   0,   0],   // прозрачный фон
//     [0,  255, 255,  80],  // бирюза
//     [0,  200, 255, 120],
//     [0,  150, 255, 160],
//     [255, 255,  0, 200],  // жёлтый
//     [255, 128,  0, 230],  // оранжевый
//     [255,   0,  0, 255]   // красный (пик)
//   ],
// });

const oceanBasemap2 = new TileLayer({
  id: 'ocean-basemap-2',
  data: 'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}',
  minZoom: 0,
  // maxZoom: 10,
  maxZoom: 5,
  tileSize: 256,

  renderSubLayers: props => {
    const {
      bbox: {west, south, east, north}
    } = props.tile;
    return new BitmapLayer(props, {
      data: null,
      image: props.data,
      bounds: [west, south, east, north]
    });
  }
});


const secondLayer = new HeatmapLayer({
  id: 'SecondHeatmapLayer',
  data: 'output.json',
  aggregation: 'SUM',
  getPosition: d => [d.lon_cell, d.lat_cell],
  getWeight: d => d.chlor_a_mean,
  radiusPixels: 45,
  intensity: 1,
  threshold: 0.11,
  weightsTextureSize: 512,
  colorRange: [
    [200, 220, 200],
    [175, 210, 165],
    [140, 190, 130],
    [100, 160, 90],
    [40, 140, 60],
    [0, 90, 30],
  ],

});

const layers = ref([oceanBasemap, secondLayer, oceanBasemap2]);

fetch('sharks.json')
  .then(response => response.json())
  .then(data => {
    // Тепер data містить масив об'єктів з координатами та числовими значеннями
    const hexagonLayer = new HexagonLayer({
      id: 'heatmap',
      colorRange: COLOR_RANGE,
      data, // тепер [{position: [lng, lat], value: N}, ...]
      elevationRange: [0, 1000],
      elevationScale: 250,
      extruded: true,
      getPosition: d => [d.lon, d.lat],   // беремо координати
      getWeight: d => 10,      // числове значення
      getColorWeight: d => 10,
      getElevationWeight: d => 10,
      elevationAggregation: 'SUM', // середнє для висоти
      colorAggregation: 'SUM',     // середнє для кольору
      coverage: 1,
      radius: 100000
    });
    layers.value = [oceanBasemap, secondLayer, hexagonLayer, oceanBasemap2];
  });

watch(layers, (val) => {
  console.log(val);
  deck.value.setProps({ layers: val })
})

deck.value = new Deck({
  parent: deckContainer.value,
  initialViewState: INITIAL_VIEW_STATE,
  controller: true,
  views: [new MapView({repeat: true})],
  layers: layers.value,
  onClick: info => {
    popUp.value.showMenu();
    coords.value = {
      x: info.x,
      y: info.y,
    };
  },
});

onBeforeUnmount(() => {
  deck.value?.finalize();
  deck.value = null;
});
</script>

<template>
  <div>
    <div ref="deckContainer"></div>
<!--    <PopUp ref="popUp" :deck="deck" :coordinates="coords"/>-->
  </div>
</template>
